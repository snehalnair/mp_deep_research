"""
Integration Guide: Evaluation Framework + MP Deep Research Agent

This script shows how to connect the evaluation framework with your
actual LangGraph agent for comprehensive evaluation.
"""

import sys
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# =============================================================================
# STEP 1: IMPORT YOUR AGENT
# =============================================================================

# Your existing agent imports
from mp_deep_research.research_agent_scope import create_research_agent
from mp_deep_research.state_scope import AgentState

# Evaluation framework imports
from mp_deep_research.evaluation import (
    # Core evaluation
    MaterialsResearchEvaluator,
    EvaluationConfig,
    EvaluationReport,
    
    # Benchmarks
    DiscoveryBenchmark,
    InnovationBenchmark,
    
    # Tool validation (Booking.com Glass Box)
    ToolSchema,
    validate_tool_call,
    check_tool_reliability,
    evaluate_tool_correctness,
    compute_tool_validation_metrics,
    
    # Consistency (τ-bench)
    evaluate_consistency,
    QueryParaphraser,
    Trajectory,
    compute_trajectory_metrics,
    analyze_complexity_scaling,
    
    # Baseline comparison
    evaluate_system,
    compare_systems,
    create_zero_shot_baseline,
    BaselineType,
    run_baseline_comparison,
)


# =============================================================================
# STEP 2: CREATE AGENT WRAPPER
# =============================================================================

class AgentWrapper:
    """
    Wrapper that adapts your LangGraph agent to the evaluation interface.
    
    The evaluation framework expects a runner function that:
    - Takes a query string
    - Returns a dict with standardized keys
    """
    
    def __init__(self, api_key: str = None):
        """Initialize the agent."""
        self.agent = create_research_agent(api_key=api_key)
        self.last_trace = None
    
    def run(self, query: str) -> Dict[str, Any]:
        """
        Run the agent and return standardized output.
        
        Args:
            query: User query string
            
        Returns:
            Dict with keys:
            - final_answer: str - The agent's response
            - success: bool - Whether task completed successfully
            - tool_calls: List[Dict] - Tools called with arguments
            - materials_found: List[str] - Material IDs mentioned
            - duration_ms: float - Execution time
            - token_count: int - Tokens used
            - reasoning_steps: List[str] - Agent's reasoning
            - error: Optional[str] - Error message if failed
        """
        start_time = time.time()
        
        try:
            # Invoke your LangGraph agent
            initial_state = {
                "messages": [{"role": "user", "content": query}],
                "materials": [],
                "current_task": None,
            }
            
            result = self.agent.invoke(initial_state)
            
            # Extract information from result
            output = self._parse_agent_result(result)
            output["duration_ms"] = (time.time() - start_time) * 1000
            output["success"] = True
            
            # Store trace for trajectory analysis
            self.last_trace = result
            
            return output
            
        except Exception as e:
            return {
                "final_answer": "",
                "success": False,
                "tool_calls": [],
                "materials_found": [],
                "duration_ms": (time.time() - start_time) * 1000,
                "token_count": 0,
                "reasoning_steps": [],
                "error": str(e),
            }
    
    def _parse_agent_result(self, result: Dict) -> Dict[str, Any]:
        """Parse LangGraph result into standardized format."""
        
        # Extract final answer from messages
        messages = result.get("messages", [])
        final_answer = ""
        for msg in reversed(messages):
            if hasattr(msg, "content") and msg.content:
                final_answer = msg.content
                break
            elif isinstance(msg, dict) and msg.get("content"):
                final_answer = msg["content"]
                break
        
        # Extract tool calls
        tool_calls = []
        reasoning_steps = []
        
        for msg in messages:
            # Check for tool calls in AIMessage
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_calls.append({
                        "name": tc.get("name", tc.get("function", {}).get("name", "")),
                        "arguments": tc.get("args", tc.get("function", {}).get("arguments", {})),
                    })
            
            # Check for additional_kwargs (OpenAI format)
            if hasattr(msg, "additional_kwargs"):
                for tc in msg.additional_kwargs.get("tool_calls", []):
                    tool_calls.append({
                        "name": tc.get("function", {}).get("name", ""),
                        "arguments": tc.get("function", {}).get("arguments", {}),
                    })
            
            # Extract reasoning from think tool
            if hasattr(msg, "name") and msg.name == "think":
                reasoning_steps.append(msg.content if hasattr(msg, "content") else str(msg))
        
        # Extract materials mentioned
        materials_found = result.get("materials", [])
        if isinstance(materials_found, list):
            materials_found = [m.get("material_id", str(m)) if isinstance(m, dict) else str(m) 
                            for m in materials_found]
        
        # Estimate token count (rough)
        token_count = sum(len(str(m).split()) * 1.3 for m in messages)
        
        return {
            "final_answer": final_answer,
            "tool_calls": tool_calls,
            "materials_found": materials_found,
            "token_count": int(token_count),
            "reasoning_steps": reasoning_steps,
        }
    
    def get_trajectory(self, task_id: str) -> Trajectory:
        """Convert last agent trace to Trajectory for analysis."""
        if not self.last_trace:
            return Trajectory(task_id=task_id)
        
        return Trajectory.from_agent_output(task_id, {
            "success": True,
            "tool_calls": self._parse_agent_result(self.last_trace)["tool_calls"],
            "final_answer": self._parse_agent_result(self.last_trace)["final_answer"],
        })


# =============================================================================
# STEP 3: DEFINE SUCCESS CHECKER
# =============================================================================

def check_task_success(output: Dict, task: Dict) -> bool:
    """
    Check if agent output successfully completes the task.
    
    Args:
        output: Agent output dict
        task: Task definition with expected results
        
    Returns:
        True if task completed successfully
    """
    if output.get("error"):
        return False
    
    if not output.get("final_answer"):
        return False
    
    # Check if expected materials were found
    expected_materials = task.get("expected_materials", [])
    if expected_materials:
        found = set(output.get("materials_found", []))
        expected = set(expected_materials)
        # Success if any expected material found
        if not (found & expected):
            return False
    
    # Check if required tools were used
    expected_tools = task.get("expected_tool_sequence", [])
    if expected_tools:
        used_tools = [tc.get("name") for tc in output.get("tool_calls", [])]
        # Success if all required tools were called
        if not set(expected_tools).issubset(set(used_tools)):
            return False
    
    return True


# =============================================================================
# STEP 4: CREATE TOOL SCHEMAS FOR VALIDATION
# =============================================================================

def get_agent_tool_schemas() -> Dict[str, ToolSchema]:
    """
    Define schemas for your agent's tools.
    Used for tool call validation.
    """
    return {
        "screen_materials": ToolSchema(
            name="screen_materials",
            description="Screen materials from the Materials Project database based on chemical composition and property criteria.",
            parameters={
                "elements": {"type": "array", "description": "List of elements to include"},
                "e_above_hull_max": {"type": "number", "description": "Maximum energy above hull (eV)"},
                "band_gap_min": {"type": "number", "description": "Minimum band gap (eV)"},
                "band_gap_max": {"type": "number", "description": "Maximum band gap (eV)"},
                "is_stable": {"type": "boolean", "description": "Filter for stable materials only"},
            },
            required_params=["elements"],
        ),
        
        "run_innovation_flow": ToolSchema(
            name="run_innovation_flow",
            description="Run the innovation workflow to design new materials via elemental substitution and ML relaxation.",
            parameters={
                "parent_material_id": {"type": "string", "description": "MP ID of parent material"},
                "substitutions": {"type": "object", "description": "Element substitution mapping"},
                "relax": {"type": "boolean", "description": "Whether to run M3GNet relaxation"},
            },
            required_params=["parent_material_id", "substitutions"],
        ),
        
        "assess_stability": ToolSchema(
            name="assess_stability",
            description="Assess thermodynamic stability of a material using convex hull analysis.",
            parameters={
                "material_id": {"type": "string", "description": "MP ID to assess"},
                "structure": {"type": "object", "description": "Structure dict if not using MP ID"},
            },
            required_params=[],
        ),
        
        "think": ToolSchema(
            name="think",
            description="Record reasoning and analysis steps. Use for planning and reflection.",
            parameters={
                "thought": {"type": "string", "description": "The reasoning content"},
            },
            required_params=["thought"],
        ),
        
        "search_arxiv": ToolSchema(
            name="search_arxiv",
            description="Search arXiv for relevant research papers on a topic.",
            parameters={
                "query": {"type": "string", "description": "Search query"},
                "max_results": {"type": "integer", "description": "Maximum papers to return"},
            },
            required_params=["query"],
        ),
    }


# =============================================================================
# STEP 5: RUN EVALUATION
# =============================================================================

def run_quick_evaluation(api_key: str = None):
    """
    Run a quick evaluation for development testing.
    """
    print("=" * 60)
    print("QUICK EVALUATION - Development Testing")
    print("=" * 60)
    
    # Initialize agent
    agent = AgentWrapper(api_key=api_key)
    
    # Get benchmarks
    discovery = DiscoveryBenchmark()
    tasks = discovery.get_tasks(difficulty="easy", limit=3)
    
    print(f"\nRunning {len(tasks)} easy tasks...")
    
    results = []
    for task in tasks:
        print(f"\n  Task: {task.task_id}")
        print(f"  Query: {task.user_query[:60]}...")
        
        output = agent.run(task.user_query)
        success = check_task_success(output, task.__dict__)
        
        results.append({
            "task_id": task.task_id,
            "success": success,
            "tools_used": len(output.get("tool_calls", [])),
            "duration_ms": output.get("duration_ms", 0),
        })
        
        print(f"  Success: {success}")
        print(f"  Tools used: {results[-1]['tools_used']}")
    
    # Summary
    success_rate = sum(r["success"] for r in results) / len(results)
    avg_tools = sum(r["tools_used"] for r in results) / len(results)
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Success Rate: {success_rate:.1%}")
    print(f"Avg Tool Calls: {avg_tools:.1f}")
    
    return results


def run_full_evaluation(api_key: str = None, output_dir: str = "./evaluation_results"):
    """
    Run full evaluation for paper submission.
    """
    print("=" * 60)
    print("FULL EVALUATION - Paper Submission")
    print("=" * 60)
    
    # Initialize agent
    agent = AgentWrapper(api_key=api_key)
    
    # Configure evaluation
    config = EvaluationConfig(
        experiment_name="MPDeepResearch_v1",
        run_discovery_benchmark=True,
        run_innovation_benchmark=True,
        use_llm_judge=True,
        judge_model="gpt-4o",
        n_bootstrap_samples=1000,
        confidence_level=0.95,
        generate_latex=True,
    )
    
    # Create evaluator
    evaluator = MaterialsResearchEvaluator(config)
    
    # Run evaluation
    print("\nRunning full benchmark suite...")
    report = evaluator.run_full_evaluation(agent_runner=agent.run)
    
    # Save results
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    report.save(f"{output_dir}/evaluation_report.json")
    
    with open(f"{output_dir}/results_table.tex", "w") as f:
        f.write(report.to_latex())
    
    with open(f"{output_dir}/results_report.md", "w") as f:
        f.write(report.to_markdown())
    
    print(f"\nResults saved to {output_dir}/")
    print(f"\nOverall Score: {report.overall_score:.3f}")
    
    return report


def run_tool_validation(api_key: str = None):
    """
    Run tool validation analysis (Booking.com Glass Box).
    """
    print("=" * 60)
    print("TOOL VALIDATION - Glass Box Analysis")
    print("=" * 60)
    
    # Get tool schemas
    tools = get_agent_tool_schemas()
    
    # Check tool reliability
    print("\n1. Tool Reliability Checks:")
    print("-" * 40)
    
    for name, schema in tools.items():
        result = check_tool_reliability(name, schema.description)
        status = "✓" if result.overall_score >= 0.7 else "✗"
        print(f"  {status} {name}: {result.overall_score:.2f}")
        if result.issues:
            for issue in result.issues[:2]:
                print(f"      - {issue}")
    
    # Run agent and validate tool calls
    print("\n2. Tool Call Validation:")
    print("-" * 40)
    
    agent = AgentWrapper(api_key=api_key)
    output = agent.run("Find stable lithium cobalt oxide cathode materials")
    
    validation_results = []
    for tc in output.get("tool_calls", []):
        result = validate_tool_call(tc, tools)
        status = "✓" if result.is_valid else "✗"
        print(f"  {status} {tc.get('name')}: {result.error_type.value}")
        validation_results.append(result)
    
    # Compute metrics
    if validation_results:
        valid_count = sum(1 for r in validation_results if r.is_valid)
        print(f"\n  Validity Rate: {valid_count}/{len(validation_results)} ({valid_count/len(validation_results):.1%})")


def run_consistency_evaluation(api_key: str = None, k: int = 3):
    """
    Run consistency evaluation (τ-bench methodology).
    """
    print("=" * 60)
    print(f"CONSISTENCY EVALUATION - pass@{k} / pass^{k}")
    print("=" * 60)
    
    agent = AgentWrapper(api_key=api_key)
    
    # Test queries
    queries = [
        "Find stable lithium battery cathode materials",
        "Search for thermoelectric materials with good ZT",
        "Identify transparent conducting oxides",
    ]
    
    print(f"\nTesting {len(queries)} queries with {k} paraphrased trials each...")
    
    def success_checker(output: Dict) -> bool:
        return output.get("success", False) and len(output.get("materials_found", [])) > 0
    
    metrics = evaluate_consistency(
        queries=queries,
        agent_runner=agent.run,
        success_checker=success_checker,
        k=k,
        seed=42,
    )
    
    print("\nResults:")
    print("-" * 40)
    print(f"  pass@{k}: {metrics.avg_pass_at_k:.3f} (any trial succeeds)")
    print(f"  pass^{k}: {metrics.avg_pass_hat_k:.3f} (ALL trials succeed)")
    print(f"  Avg Success Rate: {metrics.avg_success_rate:.3f}")
    print(f"  Fully Consistent: {metrics.fully_consistent}/{metrics.total_queries}")
    
    return metrics


def run_baseline_comparison(api_key: str = None):
    """
    Run baseline comparison (Booking.com decision framework).
    """
    print("=" * 60)
    print("BASELINE COMPARISON")
    print("=" * 60)
    
    agent = AgentWrapper(api_key=api_key)
    
    # Get benchmark tasks
    discovery = DiscoveryBenchmark()
    tasks = [
        {"id": t.task_id, "query": t.user_query, **t.__dict__}
        for t in discovery.get_tasks(limit=5)
    ]
    
    # Evaluate agent
    print("\n1. Evaluating Agent...")
    agent_eval = evaluate_system(
        system_name="MPDeepResearch Agent",
        system_type=BaselineType.AGENT,
        runner=agent.run,
        tasks=tasks,
        success_checker=check_task_success,
        model="claude-sonnet-4",
    )
    print(f"   Task Completion: {agent_eval.task_completion_rate:.1%}")
    print(f"   Avg Latency: {agent_eval.avg_latency_ms:.0f}ms")
    print(f"   Avg Tool Calls: {agent_eval.avg_tool_calls:.1f}")
    
    # Note: To run baseline comparison, you need an LLM function
    # This is a placeholder showing the structure
    print("\n2. Baseline Comparison:")
    print("   (Requires LLM API for baseline evaluation)")
    print("   See run_baseline_comparison() for implementation")
    
    return agent_eval


# =============================================================================
# STEP 6: MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="MP Deep Research Evaluation")
    parser.add_argument("--mode", choices=["quick", "full", "tools", "consistency", "baseline"],
                       default="quick", help="Evaluation mode")
    parser.add_argument("--api-key", type=str, help="API key for LLM")
    parser.add_argument("--output-dir", type=str, default="./evaluation_results",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    if args.mode == "quick":
        run_quick_evaluation(api_key=args.api_key)
    elif args.mode == "full":
        run_full_evaluation(api_key=args.api_key, output_dir=args.output_dir)
    elif args.mode == "tools":
        run_tool_validation(api_key=args.api_key)
    elif args.mode == "consistency":
        run_consistency_evaluation(api_key=args.api_key)
    elif args.mode == "baseline":
        run_baseline_comparison(api_key=args.api_key)
