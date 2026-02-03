"""
Baseline Comparison Framework

Based on Booking.com's AI Agent Evaluation methodology.
Reference: https://booking.ai/ai-agent-evaluation-82e781439d97

This module provides structured comparison of agents against simpler baselines:
1. Zero-shot LLM - Direct prompting without tools
2. Few-shot LLM - Prompting with examples
3. Deterministic Flow - Rule-based chains

Key insight from Booking.com: "Deploying an agent must be justified by a clear 
performance increase over simpler systems."
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
import time
import numpy as np


class BaselineType(Enum):
    """Types of baselines for comparison."""
    ZERO_SHOT_LLM = "zero_shot_llm"
    FEW_SHOT_LLM = "few_shot_llm"
    DETERMINISTIC_FLOW = "deterministic_flow"
    RETRIEVAL_AUGMENTED = "retrieval_augmented"
    AGENT = "agent"


@dataclass
class SystemResult:
    """Result of running a system on a single task."""
    task_id: str
    success: bool
    response: str = ""
    
    # Performance metrics
    latency_ms: float = 0.0
    token_count: int = 0
    
    # Cost estimation (in cents)
    estimated_cost_cents: float = 0.0
    
    # For agents: tool usage details
    tool_calls: List[str] = field(default_factory=list)
    num_tool_calls: int = 0
    
    # Error tracking
    error: Optional[str] = None


@dataclass
class SystemEvaluation:
    """Complete evaluation of a system across all tasks."""
    system_name: str
    system_type: BaselineType
    
    # Core metrics
    total_tasks: int = 0
    successful_tasks: int = 0
    task_completion_rate: float = 0.0
    
    # Efficiency metrics
    avg_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    avg_tokens: float = 0.0
    total_tokens: int = 0
    
    # Cost metrics
    avg_cost_cents: float = 0.0
    total_cost_cents: float = 0.0
    
    # Agent-specific
    avg_tool_calls: float = 0.0
    
    # Detailed results
    results: List[SystemResult] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "system_name": self.system_name,
            "system_type": self.system_type.value,
            "task_completion_rate": round(self.task_completion_rate, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "total_latency_ms": round(self.total_latency_ms, 2),
            "avg_tokens": round(self.avg_tokens, 2),
            "total_tokens": self.total_tokens,
            "avg_cost_cents": round(self.avg_cost_cents, 4),
            "total_cost_cents": round(self.total_cost_cents, 4),
            "avg_tool_calls": round(self.avg_tool_calls, 2),
        }


@dataclass
class ComparisonResult:
    """Result of comparing agent against baselines."""
    agent_eval: SystemEvaluation
    baseline_evals: List[SystemEvaluation] = field(default_factory=list)
    
    # Computed comparisons
    performance_deltas: Dict[str, float] = field(default_factory=dict)
    latency_ratios: Dict[str, float] = field(default_factory=dict)
    cost_ratios: Dict[str, float] = field(default_factory=dict)
    
    # Recommendations
    recommendation: str = ""
    justification: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "agent": self.agent_eval.to_dict(),
            "baselines": {b.system_name: b.to_dict() for b in self.baseline_evals},
            "performance_deltas": {k: round(v, 4) for k, v in self.performance_deltas.items()},
            "latency_ratios": {k: round(v, 2) for k, v in self.latency_ratios.items()},
            "cost_ratios": {k: round(v, 2) for k, v in self.cost_ratios.items()},
            "recommendation": self.recommendation,
            "justification": self.justification,
        }
    
    def to_latex(self) -> str:
        """Generate LaTeX comparison table."""
        lines = [
            "\\begin{table}[h]",
            "\\centering",
            "\\begin{tabular}{lccc}",
            "\\hline",
            "\\textbf{System} & \\textbf{Task Completion} & \\textbf{Latency (ms)} & \\textbf{Cost (cents)} \\\\",
            "\\hline",
        ]
        
        # Add agent row
        lines.append(
            f"\\textbf{{{self.agent_eval.system_name}}} & "
            f"{self.agent_eval.task_completion_rate:.3f} & "
            f"{self.agent_eval.avg_latency_ms:.0f} & "
            f"{self.agent_eval.avg_cost_cents:.3f} \\\\"
        )
        
        # Add baseline rows
        for baseline in self.baseline_evals:
            lines.append(
                f"{baseline.system_name} & "
                f"{baseline.task_completion_rate:.3f} & "
                f"{baseline.avg_latency_ms:.0f} & "
                f"{baseline.avg_cost_cents:.3f} \\\\"
            )
        
        lines.extend([
            "\\hline",
            "\\end{tabular}",
            f"\\caption{{System Comparison. Recommendation: {self.recommendation}}}",
            "\\label{tab:baseline_comparison}",
            "\\end{table}",
        ])
        
        return "\n".join(lines)


class CostEstimator:
    """Estimate costs for different LLM calls."""
    
    # Prices per 1M tokens (as of 2024, update as needed)
    PRICING = {
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "claude-3-opus": {"input": 15.00, "output": 75.00},
        "claude-3-sonnet": {"input": 3.00, "output": 15.00},
        "claude-3-haiku": {"input": 0.25, "output": 1.25},
        "claude-sonnet-4": {"input": 3.00, "output": 15.00},
    }
    
    def estimate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """
        Estimate cost in cents.
        
        Args:
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
        
        Returns:
            Estimated cost in cents
        """
        pricing = self.PRICING.get(model, self.PRICING["gpt-4o-mini"])
        
        input_cost = (input_tokens / 1_000_000) * pricing["input"] * 100
        output_cost = (output_tokens / 1_000_000) * pricing["output"] * 100
        
        return input_cost + output_cost


def evaluate_system(
    system_name: str,
    system_type: BaselineType,
    runner: Callable[[str], Dict],
    tasks: List[Dict[str, Any]],
    success_checker: Callable[[Dict, Dict], bool],
    model: str = "gpt-4o-mini",
) -> SystemEvaluation:
    """
    Evaluate a system on a set of tasks.
    
    Args:
        system_name: Name for this system
        system_type: Type of system (agent, baseline, etc.)
        runner: Function that takes a query and returns output dict
        tasks: List of task dicts with 'id' and 'query' keys
        success_checker: Function that checks if output is successful
        model: Model name for cost estimation
    
    Returns:
        SystemEvaluation with all metrics
    """
    evaluation = SystemEvaluation(
        system_name=system_name,
        system_type=system_type,
        total_tasks=len(tasks),
    )
    
    cost_estimator = CostEstimator()
    latencies = []
    tokens = []
    tool_calls_list = []
    
    for task in tasks:
        task_id = task.get("id", task.get("task_id", "unknown"))
        query = task.get("query", task.get("user_query", ""))
        
        start_time = time.time()
        try:
            output = runner(query)
            latency_ms = (time.time() - start_time) * 1000
            
            success = success_checker(output, task)
            
            # Extract metrics from output
            token_count = output.get("token_count", output.get("tokens", 0))
            tool_calls = output.get("tool_calls", [])
            
            # Estimate cost
            input_tokens = token_count * 0.3  # Rough estimate
            output_tokens = token_count * 0.7
            cost = cost_estimator.estimate_cost(model, input_tokens, output_tokens)
            
            result = SystemResult(
                task_id=task_id,
                success=success,
                response=output.get("final_answer", ""),
                latency_ms=latency_ms,
                token_count=token_count,
                estimated_cost_cents=cost,
                tool_calls=[tc.get("name", "") for tc in tool_calls] if isinstance(tool_calls, list) else [],
                num_tool_calls=len(tool_calls) if isinstance(tool_calls, list) else 0,
            )
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            result = SystemResult(
                task_id=task_id,
                success=False,
                latency_ms=latency_ms,
                error=str(e),
            )
        
        evaluation.results.append(result)
        
        if result.success:
            evaluation.successful_tasks += 1
        
        latencies.append(result.latency_ms)
        tokens.append(result.token_count)
        tool_calls_list.append(result.num_tool_calls)
    
    # Compute aggregates
    if tasks:
        evaluation.task_completion_rate = evaluation.successful_tasks / evaluation.total_tasks
        evaluation.avg_latency_ms = float(np.mean(latencies))
        evaluation.total_latency_ms = sum(latencies)
        evaluation.avg_tokens = float(np.mean(tokens))
        evaluation.total_tokens = sum(tokens)
        evaluation.avg_cost_cents = float(np.mean([r.estimated_cost_cents for r in evaluation.results]))
        evaluation.total_cost_cents = sum(r.estimated_cost_cents for r in evaluation.results)
        evaluation.avg_tool_calls = float(np.mean(tool_calls_list))
    
    return evaluation


def compare_systems(
    agent_eval: SystemEvaluation,
    baseline_evals: List[SystemEvaluation],
    min_performance_gain: float = 0.05,  # 5% improvement required
    max_latency_ratio: float = 3.0,       # Agent can be 3x slower max
    max_cost_ratio: float = 5.0,          # Agent can be 5x more expensive max
) -> ComparisonResult:
    """
    Compare agent against baselines and make recommendation.
    
    Based on Booking.com's decision framework:
    - Agent must show clear performance improvement
    - Cost and latency increase must be justified
    
    Args:
        agent_eval: Evaluation of the agent
        baseline_evals: Evaluations of baseline systems
        min_performance_gain: Minimum task completion improvement needed
        max_latency_ratio: Maximum acceptable latency increase
        max_cost_ratio: Maximum acceptable cost increase
    
    Returns:
        ComparisonResult with recommendation
    """
    comparison = ComparisonResult(
        agent_eval=agent_eval,
        baseline_evals=baseline_evals,
    )
    
    # Compute deltas and ratios
    best_baseline_performance = 0.0
    best_baseline_name = "none"
    
    for baseline in baseline_evals:
        name = baseline.system_name
        
        # Performance delta (positive = agent is better)
        delta = agent_eval.task_completion_rate - baseline.task_completion_rate
        comparison.performance_deltas[name] = delta
        
        # Latency ratio (>1 = agent is slower)
        if baseline.avg_latency_ms > 0:
            comparison.latency_ratios[name] = agent_eval.avg_latency_ms / baseline.avg_latency_ms
        else:
            comparison.latency_ratios[name] = 1.0
        
        # Cost ratio (>1 = agent is more expensive)
        if baseline.avg_cost_cents > 0:
            comparison.cost_ratios[name] = agent_eval.avg_cost_cents / baseline.avg_cost_cents
        else:
            comparison.cost_ratios[name] = 1.0
        
        # Track best baseline
        if baseline.task_completion_rate > best_baseline_performance:
            best_baseline_performance = baseline.task_completion_rate
            best_baseline_name = name
    
    # Make recommendation
    best_delta = comparison.performance_deltas.get(best_baseline_name, 0)
    best_latency_ratio = comparison.latency_ratios.get(best_baseline_name, 1)
    best_cost_ratio = comparison.cost_ratios.get(best_baseline_name, 1)
    
    if best_delta >= min_performance_gain:
        if best_latency_ratio <= max_latency_ratio and best_cost_ratio <= max_cost_ratio:
            comparison.recommendation = "DEPLOY_AGENT"
            comparison.justification = (
                f"Agent shows {best_delta*100:.1f}% improvement over best baseline ({best_baseline_name}) "
                f"with acceptable latency ({best_latency_ratio:.1f}x) and cost ({best_cost_ratio:.1f}x) increase."
            )
        else:
            comparison.recommendation = "OPTIMIZE_AGENT"
            comparison.justification = (
                f"Agent shows {best_delta*100:.1f}% improvement but latency ({best_latency_ratio:.1f}x) "
                f"or cost ({best_cost_ratio:.1f}x) exceeds threshold. Optimize before deployment."
            )
    else:
        comparison.recommendation = "USE_BASELINE"
        comparison.justification = (
            f"Agent improvement ({best_delta*100:.1f}%) below threshold ({min_performance_gain*100:.0f}%). "
            f"Use {best_baseline_name} for lower cost and complexity."
        )
    
    return comparison


# =============================================================================
# BASELINE IMPLEMENTATIONS
# =============================================================================

def create_zero_shot_baseline(
    llm_fn: Callable[[str], str],
    system_prompt: str = "",
) -> Callable[[str], Dict]:
    """
    Create a zero-shot LLM baseline runner.
    
    Args:
        llm_fn: Function that takes a prompt and returns LLM response
        system_prompt: Optional system prompt
    
    Returns:
        Runner function compatible with evaluate_system()
    """
    def runner(query: str) -> Dict:
        prompt = query
        if system_prompt:
            prompt = f"{system_prompt}\n\n{query}"
        
        start_time = time.time()
        response = llm_fn(prompt)
        duration = (time.time() - start_time) * 1000
        
        return {
            "final_answer": response,
            "duration_ms": duration,
            "token_count": len(response.split()) * 1.3,  # Rough estimate
            "tool_calls": [],
        }
    
    return runner


def create_few_shot_baseline(
    llm_fn: Callable[[str], str],
    examples: List[Dict[str, str]],
    system_prompt: str = "",
) -> Callable[[str], Dict]:
    """
    Create a few-shot LLM baseline runner.
    
    Args:
        llm_fn: Function that takes a prompt and returns LLM response
        examples: List of {"query": ..., "response": ...} examples
        system_prompt: Optional system prompt
    
    Returns:
        Runner function compatible with evaluate_system()
    """
    def runner(query: str) -> Dict:
        # Build few-shot prompt
        prompt_parts = []
        
        if system_prompt:
            prompt_parts.append(system_prompt)
        
        prompt_parts.append("Here are some examples:")
        for ex in examples:
            prompt_parts.append(f"\nQuery: {ex['query']}\nResponse: {ex['response']}")
        
        prompt_parts.append(f"\nNow respond to this query:\nQuery: {query}\nResponse:")
        
        prompt = "\n".join(prompt_parts)
        
        start_time = time.time()
        response = llm_fn(prompt)
        duration = (time.time() - start_time) * 1000
        
        return {
            "final_answer": response,
            "duration_ms": duration,
            "token_count": len(prompt.split()) * 1.3 + len(response.split()) * 1.3,
            "tool_calls": [],
        }
    
    return runner


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def run_baseline_comparison(
    agent_runner: Callable[[str], Dict],
    tasks: List[Dict[str, Any]],
    success_checker: Callable[[Dict, Dict], bool],
    llm_fn: Callable[[str], str] = None,
    examples: List[Dict[str, str]] = None,
    agent_name: str = "MPDeepResearch Agent",
    model: str = "claude-sonnet-4",
) -> ComparisonResult:
    """
    Run complete baseline comparison.
    
    Args:
        agent_runner: Agent's runner function
        tasks: List of tasks to evaluate
        success_checker: Function to check task success
        llm_fn: Optional LLM function for baselines
        examples: Optional examples for few-shot baseline
        agent_name: Name of the agent
        model: Model name for cost estimation
    
    Returns:
        ComparisonResult with full comparison
    """
    # Evaluate agent
    agent_eval = evaluate_system(
        system_name=agent_name,
        system_type=BaselineType.AGENT,
        runner=agent_runner,
        tasks=tasks,
        success_checker=success_checker,
        model=model,
    )
    
    baseline_evals = []
    
    # Evaluate baselines if LLM function provided
    if llm_fn:
        # Zero-shot baseline
        zero_shot_runner = create_zero_shot_baseline(
            llm_fn,
            system_prompt="You are a materials science expert. Answer the following query."
        )
        zero_shot_eval = evaluate_system(
            system_name="Zero-shot LLM",
            system_type=BaselineType.ZERO_SHOT_LLM,
            runner=zero_shot_runner,
            tasks=tasks,
            success_checker=success_checker,
            model=model,
        )
        baseline_evals.append(zero_shot_eval)
        
        # Few-shot baseline (if examples provided)
        if examples:
            few_shot_runner = create_few_shot_baseline(
                llm_fn,
                examples,
                system_prompt="You are a materials science expert."
            )
            few_shot_eval = evaluate_system(
                system_name="Few-shot LLM",
                system_type=BaselineType.FEW_SHOT_LLM,
                runner=few_shot_runner,
                tasks=tasks,
                success_checker=success_checker,
                model=model,
            )
            baseline_evals.append(few_shot_eval)
    
    # Compare
    return compare_systems(agent_eval, baseline_evals)
