"""
Consistency and Trajectory Evaluation

Based on Booking.com's AI Agent Evaluation and τ-bench research.
Reference: https://booking.ai/ai-agent-evaluation-82e781439d97

This module provides:
1. Consistency Evaluation - pass@k and pass^k metrics with paraphrased queries
2. Trajectory Optimality - Evaluating if agent takes optimal paths
3. Complexity Scaling Analysis - How performance degrades with task complexity

References:
- τ-bench: A Benchmark for Tool-Agent-User Interaction (Yao et al., 2024)
- Apigen-mt: Agentic pipeline for multi-turn data generation (Prabhakar et al., 2025)
- GAIA: A benchmark for General AI Assistants (Mialon et al., 2023)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Tuple
from collections import defaultdict
import numpy as np
import time
import random


# =============================================================================
# CONSISTENCY EVALUATION
# =============================================================================

@dataclass
class ConsistencyTrialResult:
    """Result of a single trial (one query variant)."""
    query: str
    success: bool
    response: Optional[str] = None
    tool_calls: List[str] = field(default_factory=list)
    duration_ms: float = 0.0
    error: Optional[str] = None


@dataclass 
class ConsistencyResult:
    """Consistency result for a single original query tested with k variants."""
    original_query: str
    trials: List[ConsistencyTrialResult] = field(default_factory=list)
    
    @property
    def k(self) -> int:
        """Number of trials."""
        return len(self.trials)
    
    @property
    def successes(self) -> int:
        """Number of successful trials."""
        return sum(1 for t in self.trials if t.success)
    
    @property
    def pass_at_k(self) -> float:
        """
        Probability that at least one of k trials succeeds.
        This is the optimistic metric - the agent CAN solve the task.
        """
        return 1.0 if any(t.success for t in self.trials) else 0.0
    
    @property
    def pass_hat_k(self) -> float:
        """
        Probability that ALL k trials succeed.
        This is the consistency metric - the agent RELIABLY solves the task.
        Also called "pass^k" in τ-bench paper.
        """
        return 1.0 if all(t.success for t in self.trials) else 0.0
    
    @property
    def success_rate(self) -> float:
        """Fraction of successful trials."""
        return self.successes / self.k if self.k > 0 else 0.0
    
    def to_dict(self) -> Dict:
        return {
            "original_query": self.original_query,
            "k": self.k,
            "successes": self.successes,
            "pass@k": self.pass_at_k,
            "pass^k": self.pass_hat_k,
            "success_rate": round(self.success_rate, 4),
        }


@dataclass
class ConsistencyMetrics:
    """Aggregate consistency metrics across all queries."""
    total_queries: int = 0
    k: int = 0
    
    # Core metrics from τ-bench
    avg_pass_at_k: float = 0.0   # Average probability of at least one success
    avg_pass_hat_k: float = 0.0  # Average probability of all successes (consistency)
    avg_success_rate: float = 0.0
    
    # Distribution analysis
    fully_consistent: int = 0    # Queries where all k trials succeeded
    fully_inconsistent: int = 0  # Queries where all k trials failed
    partial_success: int = 0     # Queries with mixed results
    
    # Detailed results
    results: List[ConsistencyResult] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "total_queries": self.total_queries,
            "k_trials": self.k,
            "avg_pass@k": round(self.avg_pass_at_k, 4),
            "avg_pass^k": round(self.avg_pass_hat_k, 4),
            "avg_success_rate": round(self.avg_success_rate, 4),
            "fully_consistent": self.fully_consistent,
            "fully_inconsistent": self.fully_inconsistent,
            "partial_success": self.partial_success,
            "consistency_rate": round(self.fully_consistent / max(1, self.total_queries), 4),
        }
    
    def to_latex(self) -> str:
        """Generate LaTeX table rows for paper."""
        return f"""\\textbf{{Consistency}} & pass@{self.k} & {self.avg_pass_at_k:.3f} \\\\
 & pass$^{{{self.k}}}$ & {self.avg_pass_hat_k:.3f} \\\\
 & Success Rate & {self.avg_success_rate:.3f} \\\\
 & Consistency Rate & {self.fully_consistent / max(1, self.total_queries):.3f} \\\\"""


class QueryParaphraser:
    """
    Generate paraphrased versions of queries for consistency testing.
    
    Supports:
    1. Rule-based paraphrasing (default)
    2. LLM-based paraphrasing (optional)
    """
    
    # Synonym substitutions for materials science
    SYNONYMS = {
        "find": ["search for", "locate", "identify", "discover"],
        "materials": ["compounds", "substances", "structures"],
        "stable": ["thermodynamically stable", "energetically favorable"],
        "create": ["design", "generate", "synthesize", "develop"],
        "substitute": ["replace", "swap", "exchange"],
        "analyze": ["examine", "evaluate", "assess", "investigate"],
        "compare": ["contrast", "evaluate", "benchmark"],
        "recommend": ["suggest", "propose", "identify"],
        "good": ["promising", "favorable", "suitable"],
        "battery": ["energy storage", "electrochemical cell"],
        "cathode": ["positive electrode", "cathode material"],
    }
    
    # Prefix variations
    PREFIXES = [
        "",
        "Please ",
        "Can you ",
        "Could you ",
        "I'd like you to ",
        "Help me ",
        "I need to ",
    ]
    
    # Suffix variations
    SUFFIXES = [
        "",
        " please",
        " thanks",
        " for my research",
        " if possible",
    ]
    
    def paraphrase(self, query: str, k: int = 5, seed: int = None) -> List[str]:
        """
        Generate k paraphrased versions of a query.
        
        Args:
            query: Original query
            k: Number of paraphrases to generate
            seed: Random seed for reproducibility
        
        Returns:
            List of k paraphrased queries (always includes original)
        """
        if seed is not None:
            random.seed(seed)
        
        paraphrases = [query]  # Always include original
        query_lower = query.lower()
        
        # Generate synonym-based paraphrases
        for word, synonyms in self.SYNONYMS.items():
            if len(paraphrases) >= k:
                break
            if word in query_lower:
                for synonym in synonyms:
                    if len(paraphrases) >= k:
                        break
                    new_query = query.replace(word, synonym).replace(word.capitalize(), synonym.capitalize())
                    if new_query not in paraphrases:
                        paraphrases.append(new_query)
        
        # Generate prefix/suffix variations
        for prefix in self.PREFIXES:
            if len(paraphrases) >= k:
                break
            for suffix in self.SUFFIXES:
                if len(paraphrases) >= k:
                    break
                # Adjust query for prefix
                modified = query
                if prefix and modified[0].isupper():
                    modified = modified[0].lower() + modified[1:]
                new_query = prefix + modified + suffix
                if new_query not in paraphrases and new_query != query:
                    paraphrases.append(new_query)
        
        return paraphrases[:k]
    
    def paraphrase_with_llm(
        self,
        query: str,
        k: int = 5,
        llm_fn: Callable[[str], str] = None,
    ) -> List[str]:
        """
        Generate paraphrases using an LLM.
        
        Args:
            query: Original query
            k: Number of paraphrases
            llm_fn: Function that takes a prompt and returns LLM response
        
        Returns:
            List of paraphrased queries
        """
        if llm_fn is None:
            return self.paraphrase(query, k)
        
        prompt = f"""Generate {k-1} paraphrased versions of this query. 
Each paraphrase should have the same meaning but different wording.
Return only the paraphrases, one per line.

Original query: {query}

Paraphrases:"""
        
        try:
            response = llm_fn(prompt)
            paraphrases = [query]  # Include original
            for line in response.strip().split('\n'):
                line = line.strip().lstrip('0123456789.-) ')
                if line and line not in paraphrases:
                    paraphrases.append(line)
            return paraphrases[:k]
        except Exception:
            return self.paraphrase(query, k)


def evaluate_consistency(
    queries: List[str],
    agent_runner: Callable[[str], Dict],
    success_checker: Callable[[Dict], bool],
    k: int = 5,
    paraphraser: QueryParaphraser = None,
    seed: int = 42,
) -> ConsistencyMetrics:
    """
    Evaluate agent consistency using paraphrased queries.
    
    This implements the τ-bench consistency evaluation methodology:
    - For each query, generate k paraphrased versions
    - Run agent on each paraphrase
    - Compute pass@k (any success) and pass^k (all success)
    
    Low pass^k indicates the agent is inconsistent - it can solve tasks
    but not reliably.
    
    Args:
        queries: List of original queries to test
        agent_runner: Function that runs agent and returns output dict
        success_checker: Function that checks if output is successful
        k: Number of paraphrased trials per query
        paraphraser: Optional custom paraphraser
        seed: Random seed for reproducibility
    
    Returns:
        ConsistencyMetrics with all results
    """
    if paraphraser is None:
        paraphraser = QueryParaphraser()
    
    metrics = ConsistencyMetrics(total_queries=len(queries), k=k)
    
    pass_at_k_scores = []
    pass_hat_k_scores = []
    success_rates = []
    
    for i, query in enumerate(queries):
        # Generate paraphrases
        paraphrases = paraphraser.paraphrase(query, k, seed=seed + i)
        
        # Run trials
        trials = []
        for paraphrase in paraphrases:
            start_time = time.time()
            try:
                output = agent_runner(paraphrase)
                success = success_checker(output)
                trial = ConsistencyTrialResult(
                    query=paraphrase,
                    success=success,
                    response=output.get("final_answer"),
                    tool_calls=output.get("tool_calls", []),
                    duration_ms=(time.time() - start_time) * 1000,
                )
            except Exception as e:
                trial = ConsistencyTrialResult(
                    query=paraphrase,
                    success=False,
                    error=str(e),
                    duration_ms=(time.time() - start_time) * 1000,
                )
            trials.append(trial)
        
        # Create result
        result = ConsistencyResult(original_query=query, trials=trials)
        metrics.results.append(result)
        
        # Update scores
        pass_at_k_scores.append(result.pass_at_k)
        pass_hat_k_scores.append(result.pass_hat_k)
        success_rates.append(result.success_rate)
        
        # Categorize
        if result.pass_hat_k == 1.0:
            metrics.fully_consistent += 1
        elif result.pass_at_k == 0.0:
            metrics.fully_inconsistent += 1
        else:
            metrics.partial_success += 1
    
    # Compute averages
    if queries:
        metrics.avg_pass_at_k = float(np.mean(pass_at_k_scores))
        metrics.avg_pass_hat_k = float(np.mean(pass_hat_k_scores))
        metrics.avg_success_rate = float(np.mean(success_rates))
    
    return metrics


# =============================================================================
# TRAJECTORY OPTIMALITY
# =============================================================================

@dataclass
class TrajectoryStep:
    """A single step in an agent trajectory."""
    step_number: int
    action_type: str  # "tool_call", "reasoning", "response"
    tool_name: Optional[str] = None
    tool_args: Optional[Dict] = None
    duration_ms: float = 0.0
    tokens_used: int = 0


@dataclass
class Trajectory:
    """Complete trajectory of an agent execution."""
    task_id: str
    steps: List[TrajectoryStep] = field(default_factory=list)
    success: bool = False
    final_answer: str = ""
    total_duration_ms: float = 0.0
    total_tokens: int = 0
    
    @property
    def num_steps(self) -> int:
        return len(self.steps)
    
    @property
    def num_tool_calls(self) -> int:
        return sum(1 for s in self.steps if s.action_type == "tool_call")
    
    @property
    def tool_sequence(self) -> List[str]:
        return [s.tool_name for s in self.steps if s.tool_name]
    
    @classmethod
    def from_agent_output(cls, task_id: str, output: Dict) -> "Trajectory":
        """Create Trajectory from agent output dict."""
        steps = []
        for i, tc in enumerate(output.get("tool_calls", [])):
            steps.append(TrajectoryStep(
                step_number=i,
                action_type="tool_call",
                tool_name=tc.get("name"),
                tool_args=tc.get("arguments"),
            ))
        
        return cls(
            task_id=task_id,
            steps=steps,
            success=output.get("success", False),
            final_answer=output.get("final_answer", ""),
            total_duration_ms=output.get("duration_ms", 0.0),
            total_tokens=output.get("token_count", 0),
        )


@dataclass
class TrajectoryOptimalityResult:
    """Result of trajectory optimality evaluation."""
    task_id: str
    actual_trajectory: Trajectory
    optimal_steps: Optional[int] = None
    optimal_tools: Optional[List[str]] = None
    
    # Computed metrics
    is_optimal: bool = False
    step_efficiency: float = 0.0
    tool_efficiency: float = 0.0
    extra_tool_calls: int = 0
    missing_tools: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "task_id": self.task_id,
            "is_optimal": self.is_optimal,
            "actual_steps": self.actual_trajectory.num_steps,
            "actual_tool_calls": self.actual_trajectory.num_tool_calls,
            "optimal_steps": self.optimal_steps,
            "step_efficiency": round(self.step_efficiency, 4),
            "tool_efficiency": round(self.tool_efficiency, 4),
            "extra_tool_calls": self.extra_tool_calls,
        }


def evaluate_trajectory_optimality(
    trajectory: Trajectory,
    optimal_steps: int = None,
    optimal_tool_sequence: List[str] = None,
) -> TrajectoryOptimalityResult:
    """
    Evaluate if a trajectory is optimal.
    
    Three evaluation modes:
    1. Step count comparison (if optimal_steps provided)
    2. Tool sequence comparison (if optimal_tool_sequence provided)
    3. Heuristic (fewer steps = better)
    
    Args:
        trajectory: The agent's actual trajectory
        optimal_steps: Expected number of steps for optimal solution
        optimal_tool_sequence: Expected sequence of tool calls
    
    Returns:
        TrajectoryOptimalityResult
    """
    result = TrajectoryOptimalityResult(
        task_id=trajectory.task_id,
        actual_trajectory=trajectory,
        optimal_steps=optimal_steps,
        optimal_tools=optimal_tool_sequence,
    )
    
    actual_tools = trajectory.tool_sequence
    actual_steps = trajectory.num_steps
    
    if optimal_tool_sequence is not None:
        # Compare tool sequences
        optimal_set = set(optimal_tool_sequence)
        actual_set = set(actual_tools)
        
        result.missing_tools = list(optimal_set - actual_set)
        extra = actual_set - optimal_set
        result.extra_tool_calls = len(extra)
        
        # Tool efficiency
        if len(actual_tools) > 0:
            result.tool_efficiency = min(1.0, len(optimal_tool_sequence) / len(actual_tools))
        else:
            result.tool_efficiency = 0.0 if optimal_tool_sequence else 1.0
        
        # Optimal if all expected tools used with no extras
        result.is_optimal = (
            len(result.missing_tools) == 0 and 
            result.extra_tool_calls == 0
        )
        
        result.step_efficiency = result.tool_efficiency
        
    elif optimal_steps is not None:
        # Compare step counts
        result.step_efficiency = min(1.0, optimal_steps / max(1, actual_steps))
        result.tool_efficiency = result.step_efficiency
        result.is_optimal = actual_steps <= optimal_steps
        result.extra_tool_calls = max(0, actual_steps - optimal_steps)
        
    else:
        # Heuristic: assume 3-5 tool calls is optimal for most tasks
        baseline = 4
        result.step_efficiency = min(1.0, baseline / max(1, trajectory.num_tool_calls))
        result.tool_efficiency = result.step_efficiency
        result.is_optimal = trajectory.num_tool_calls <= baseline
        result.extra_tool_calls = max(0, trajectory.num_tool_calls - baseline)
    
    return result


@dataclass
class TrajectoryMetrics:
    """Aggregate trajectory optimality metrics."""
    total_tasks: int = 0
    optimal_count: int = 0
    
    avg_step_efficiency: float = 0.0
    avg_tool_efficiency: float = 0.0
    avg_tool_calls: float = 0.0
    avg_duration_ms: float = 0.0
    
    # Success comparison
    avg_tools_on_success: float = 0.0
    avg_tools_on_failure: float = 0.0
    
    results: List[TrajectoryOptimalityResult] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "total_tasks": self.total_tasks,
            "optimal_count": self.optimal_count,
            "optimality_rate": round(self.optimal_count / max(1, self.total_tasks), 4),
            "avg_step_efficiency": round(self.avg_step_efficiency, 4),
            "avg_tool_efficiency": round(self.avg_tool_efficiency, 4),
            "avg_tool_calls": round(self.avg_tool_calls, 2),
            "avg_duration_ms": round(self.avg_duration_ms, 2),
            "avg_tools_on_success": round(self.avg_tools_on_success, 2),
            "avg_tools_on_failure": round(self.avg_tools_on_failure, 2),
        }


def compute_trajectory_metrics(
    trajectories: List[Trajectory],
    optimal_sequences: Dict[str, List[str]] = None,
    optimal_step_counts: Dict[str, int] = None,
) -> TrajectoryMetrics:
    """
    Compute trajectory metrics across multiple tasks.
    
    Args:
        trajectories: List of agent trajectories
        optimal_sequences: Dict mapping task_id to optimal tool sequence
        optimal_step_counts: Dict mapping task_id to optimal step count
    
    Returns:
        TrajectoryMetrics
    """
    metrics = TrajectoryMetrics(total_tasks=len(trajectories))
    
    step_efficiencies = []
    tool_efficiencies = []
    tool_counts = []
    durations = []
    
    success_tools = []
    failure_tools = []
    
    for traj in trajectories:
        opt_seq = optimal_sequences.get(traj.task_id) if optimal_sequences else None
        opt_steps = optimal_step_counts.get(traj.task_id) if optimal_step_counts else None
        
        result = evaluate_trajectory_optimality(traj, opt_steps, opt_seq)
        metrics.results.append(result)
        
        if result.is_optimal:
            metrics.optimal_count += 1
        
        step_efficiencies.append(result.step_efficiency)
        tool_efficiencies.append(result.tool_efficiency)
        tool_counts.append(traj.num_tool_calls)
        durations.append(traj.total_duration_ms)
        
        if traj.success:
            success_tools.append(traj.num_tool_calls)
        else:
            failure_tools.append(traj.num_tool_calls)
    
    if trajectories:
        metrics.avg_step_efficiency = float(np.mean(step_efficiencies))
        metrics.avg_tool_efficiency = float(np.mean(tool_efficiencies))
        metrics.avg_tool_calls = float(np.mean(tool_counts))
        metrics.avg_duration_ms = float(np.mean(durations))
    
    if success_tools:
        metrics.avg_tools_on_success = float(np.mean(success_tools))
    if failure_tools:
        metrics.avg_tools_on_failure = float(np.mean(failure_tools))
    
    return metrics


# =============================================================================
# COMPLEXITY SCALING ANALYSIS
# =============================================================================

@dataclass
class ComplexityBucket:
    """Results for a single complexity level."""
    level: int
    name: str  # "easy", "medium", "hard"
    task_count: int = 0
    success_count: int = 0
    avg_tool_calls: float = 0.0
    avg_duration_ms: float = 0.0
    
    @property
    def success_rate(self) -> float:
        return self.success_count / max(1, self.task_count)


@dataclass
class ComplexityAnalysis:
    """Analysis of how performance degrades with task complexity."""
    buckets: Dict[int, ComplexityBucket] = field(default_factory=dict)
    
    # Degradation metrics
    performance_slope: float = 0.0  # Negative = degrades with complexity
    tool_calls_slope: float = 0.0   # Positive = more tools for harder tasks
    
    def to_dict(self) -> Dict:
        return {
            "buckets": {
                str(level): {
                    "name": b.name,
                    "task_count": b.task_count,
                    "success_rate": round(b.success_rate, 4),
                    "avg_tool_calls": round(b.avg_tool_calls, 2),
                }
                for level, b in self.buckets.items()
            },
            "performance_degradation_slope": round(self.performance_slope, 4),
            "tool_calls_growth_slope": round(self.tool_calls_slope, 4),
        }


def analyze_complexity_scaling(
    results: List[Tuple[int, bool, int, float]],  # (complexity, success, tools, duration)
    level_names: Dict[int, str] = None,
) -> ComplexityAnalysis:
    """
    Analyze how agent performance scales with task complexity.
    
    Based on GAIA benchmark methodology - good agents should:
    - Maintain high success rate as complexity increases
    - Scale tool usage appropriately
    
    Args:
        results: List of (complexity_level, success, tool_calls, duration_ms) tuples
        level_names: Optional dict mapping level numbers to names
    
    Returns:
        ComplexityAnalysis
    """
    if level_names is None:
        level_names = {1: "easy", 2: "medium", 3: "hard"}
    
    analysis = ComplexityAnalysis()
    
    # Group by complexity level
    by_level: Dict[int, List] = defaultdict(list)
    for complexity, success, tools, duration in results:
        by_level[complexity].append((success, tools, duration))
    
    # Compute per-level metrics
    levels = sorted(by_level.keys())
    success_rates = []
    tool_counts = []
    
    for level in levels:
        data = by_level[level]
        
        bucket = ComplexityBucket(
            level=level,
            name=level_names.get(level, f"level_{level}"),
            task_count=len(data),
            success_count=sum(1 for s, _, _ in data if s),
            avg_tool_calls=float(np.mean([t for _, t, _ in data])),
            avg_duration_ms=float(np.mean([d for _, _, d in data])),
        )
        analysis.buckets[level] = bucket
        
        success_rates.append((level, bucket.success_rate))
        tool_counts.append((level, bucket.avg_tool_calls))
    
    # Compute slopes (linear regression)
    if len(levels) >= 2:
        x = np.array([sr[0] for sr in success_rates])
        y_success = np.array([sr[1] for sr in success_rates])
        y_tools = np.array([tc[1] for tc in tool_counts])
        
        # Performance slope (negative = degradation)
        analysis.performance_slope = float(np.polyfit(x, y_success, 1)[0])
        
        # Tool calls slope (positive = more tools for harder tasks)
        analysis.tool_calls_slope = float(np.polyfit(x, y_tools, 1)[0])
    
    return analysis
