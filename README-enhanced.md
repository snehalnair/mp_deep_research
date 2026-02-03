# Materials Project Deep Research - Evaluation Framework

## Overview

A comprehensive evaluation framework for the **Materials Project Deep Research** agent system, enhanced with **Booking.com's AI Agent Evaluation** best practices.

This framework provides rigorous evaluation capabilities for:
1. **Materials Science Metrics** - Screening, Innovation, Stability accuracy
2. **Agent Performance** - Task completion, reasoning quality
3. **Tool Validation** - Call validity, correctness, reliability (Booking.com Glass Box)
4. **Consistency** - pass@k and pass^k metrics (τ-bench methodology)
5. **Baseline Comparison** - Agent vs simpler systems decision framework

## Architecture

```
evaluation/
├── __init__.py          # Package exports
├── metrics.py           # Core metrics (screening, innovation, stability)
├── benchmarks.py        # Discovery & Innovation benchmark tasks
├── datasets.py          # Ground truth dataset generation
├── evaluator.py         # Main evaluator with LLM-as-Judge
├── tool_validation.py   # NEW: Tool validity, correctness, reliability
├── consistency.py       # NEW: pass@k, pass^k, trajectory optimality
└── baselines.py         # NEW: Baseline comparison framework

tests/
├── test_evaluation.py        # Core module tests
└── test_booking_evaluation.py # New module tests
```

---

## Key Features

### 1. Tool Validation (Glass Box Evaluation)

Based on **Booking.com's Glass Box evaluation** and **Berkeley Function Calling Leaderboard**.

```python
from evaluation import validate_tool_call, check_tool_reliability, ToolSchema

# Define tool schema
tools = {
    "screen_materials": ToolSchema(
        name="screen_materials",
        description="Screen materials based on stability criteria...",
        parameters={"elements": {"type": "array"}, ...},
        required_params=["elements"],
    )
}

# Validate a tool call
result = validate_tool_call(
    {"name": "screen_materials", "arguments": {"elements": ["Li", "Co", "O"]}},
    tools
)
print(f"Valid: {result.is_valid}")  # True

# Check tool reliability
reliability = check_tool_reliability(
    tool_name="screen_materials_by_stability",
    tool_description="Screen materials from database..."
)
print(f"Reliability Score: {reliability.overall_score:.2f}")  # 0.91
```

**Checks performed:**
- ✅ Tool name exists
- ✅ Required parameters present
- ✅ No unexpected parameters  
- ✅ Parameter types match schema
- ✅ Tool name uses snake_case, starts with verb
- ✅ Description not too short/long
- ✅ Max 3 optional arguments

### 2. Consistency Evaluation (τ-bench)

Based on **τ-bench research** - measures how reliably an agent solves tasks.

```python
from evaluation import evaluate_consistency, QueryParaphraser

# Generate paraphrased queries
paraphraser = QueryParaphraser()
paraphrases = paraphraser.paraphrase("Find stable Li-ion cathodes", k=5)
# ['Find stable Li-ion cathodes',
#  'Search for stable Li-ion cathodes', 
#  'Locate stable Li-ion cathodes', ...]

# Evaluate consistency
metrics = evaluate_consistency(
    queries=["Find cathodes", "Design novel material"],
    agent_runner=my_agent,
    success_checker=check_success,
    k=5,  # 5 paraphrased trials per query
)

print(f"pass@5: {metrics.avg_pass_at_k:.3f}")   # Any trial succeeds
print(f"pass^5: {metrics.avg_pass_hat_k:.3f}")  # ALL trials succeed (consistency)
```

**Key insight**: Low pass^k indicates the agent is **inconsistent** - it can solve tasks but not reliably.

### 3. Trajectory Optimality

Evaluate if the agent takes optimal paths to solve tasks.

```python
from evaluation import Trajectory, evaluate_trajectory_optimality

trajectory = Trajectory.from_agent_output("task_001", agent_output)

result = evaluate_trajectory_optimality(
    trajectory,
    optimal_tool_sequence=["screen_materials", "think"],
)

print(f"Optimal: {result.is_optimal}")
print(f"Tool Efficiency: {result.tool_efficiency:.2f}")
print(f"Extra Tool Calls: {result.extra_tool_calls}")
```

### 4. Baseline Comparison

Based on **Booking.com's decision framework**: "Deploying an agent must be justified by a clear performance increase over simpler systems."

```python
from evaluation import (
    evaluate_system, compare_systems, 
    create_zero_shot_baseline, BaselineType
)

# Evaluate agent
agent_eval = evaluate_system(
    system_name="MPDeepResearch Agent",
    system_type=BaselineType.AGENT,
    runner=agent_runner,
    tasks=tasks,
    success_checker=success_checker,
)

# Evaluate zero-shot baseline
baseline_runner = create_zero_shot_baseline(
    llm_fn=call_llm,
    system_prompt="You are a materials science expert."
)
baseline_eval = evaluate_system(
    system_name="Zero-shot LLM",
    system_type=BaselineType.ZERO_SHOT_LLM,
    runner=baseline_runner,
    tasks=tasks,
    success_checker=success_checker,
)

# Compare and get recommendation
comparison = compare_systems(
    agent_eval,
    [baseline_eval],
    min_performance_gain=0.05,  # Need 5% improvement
    max_latency_ratio=3.0,      # Can be 3x slower
)

print(comparison.recommendation)  # "DEPLOY_AGENT" or "USE_BASELINE"
print(comparison.justification)
```

**Output:**
```
DEPLOY_AGENT
Agent shows 25.0% improvement over best baseline (Zero-shot LLM) 
with acceptable latency (2.5x) and cost (3.2x) increase.
```

### 5. Complexity Scaling Analysis

Analyze how performance degrades with task complexity (GAIA methodology).

```python
from evaluation import analyze_complexity_scaling

results = [
    # (complexity_level, success, tool_calls, duration_ms)
    (1, True, 2, 100),   # Easy
    (1, True, 2, 110),
    (2, True, 3, 200),   # Medium  
    (2, False, 4, 250),
    (3, False, 5, 400),  # Hard
    (3, False, 6, 450),
]

analysis = analyze_complexity_scaling(results)

print(f"Easy success rate: {analysis.buckets[1].success_rate:.2f}")
print(f"Hard success rate: {analysis.buckets[3].success_rate:.2f}")
print(f"Performance slope: {analysis.performance_slope:.4f}")  # Negative = degradation
```

---

## Metrics Summary for Paper

| Category | Metric | Description |
|----------|--------|-------------|
| **Discovery** | Precision/Recall/F1 | Classification accuracy |
| | NDCG@10 | Ranking quality |
| | MRR | First relevant result |
| **Innovation** | Energy MAE (meV) | M3GNet vs DFT accuracy |
| | Stability F1 | Stability prediction |
| | Success Rate | Confirmed/Predicted |
| **Tool Validation** | Validity Rate | Valid tool calls |
| | Correctness Rate | Right tools used |
| | Reliability Score | Tool definition quality |
| **Consistency** | pass@k | Any trial succeeds |
| | pass^k | All trials succeed |
| | Consistency Rate | Fully consistent queries |
| **Efficiency** | Trajectory Optimality | Optimal path taken |
| | Tool Efficiency | Min tools used |
| | Latency/Cost | vs baseline comparison |

---

## Quick Start

### Development Testing
```python
from evaluation import MaterialsResearchEvaluator, EvaluationConfig

config = EvaluationConfig(
    experiment_name="dev_test",
    run_discovery_benchmark=True,
    use_llm_judge=False,  # Faster
)

evaluator = MaterialsResearchEvaluator(config)
report = evaluator.run_full_evaluation(agent_runner=my_agent)

print(f"Task Completion: {report.agent_metrics.task_completion_rate:.2%}")
```

### Full Paper Evaluation
```python
config = EvaluationConfig(
    experiment_name="MPDeepResearch_Paper_v1",
    run_discovery_benchmark=True,
    run_innovation_benchmark=True,
    use_llm_judge=True,
    judge_model="gpt-4o",
    n_bootstrap_samples=1000,
    generate_latex=True,
)

evaluator = MaterialsResearchEvaluator(config)
report = evaluator.run_full_evaluation(agent_runner=my_agent)

# Generate LaTeX tables
print(report.to_latex())

# Save JSON
report.save("results/evaluation_report.json")
```

---

## References

1. **Booking.com AI Agent Evaluation** (2026)
   - https://booking.ai/ai-agent-evaluation-82e781439d97
   
2. **τ-bench**: A Benchmark for Tool-Agent-User Interaction
   - Yao et al., 2024
   
3. **GAIA**: A benchmark for General AI Assistants
   - Mialon et al., 2023
   
4. **Berkeley Function Calling Leaderboard**
   - https://gorilla.cs.berkeley.edu/blogs/8_berkeley_function_calling_leaderboard.html

5. **EASYTOOL**: Enhancing LLM-Based Agents with Concise Tool Instruction
   - Yuan et al., 2024

---

## Integration with Your Agent

```python
def my_agent_runner(query: str) -> Dict:
    """
    Wrap your LangGraph agent for evaluation.
    
    Returns dict with keys:
    - final_answer: str
    - success: bool
    - tool_calls: List[Dict]  
    - duration_ms: float
    - token_count: int
    """
    from your_agent import research_agent
    
    result = research_agent.invoke({
        "messages": [{"role": "user", "content": query}]
    })
    
    return {
        "final_answer": result["final_response"],
        "success": result.get("success", True),
        "tool_calls": extract_tool_calls(result),
        "duration_ms": result.get("duration", 0) * 1000,
        "token_count": result.get("tokens", 0),
    }
```

---

## Running Tests

```bash
cd mp_deep_research
PYTHONPATH=./src pytest tests/test_evaluation.py tests/test_booking_evaluation.py -v
```

---

## License

MIT License
