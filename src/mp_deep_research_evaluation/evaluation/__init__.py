"""
Materials Project Deep Research - Evaluation Framework

A comprehensive evaluation system for assessing:
1. Tool-level accuracy (screening, innovation, stability)
2. Agent-level performance (reasoning, planning, tool selection)
3. End-to-end benchmark results (discovery & innovation tasks)
4. Scientific validity metrics

Enhanced with Booking.com's AI Agent Evaluation best practices:
5. Tool validation (validity, correctness, reliability)
6. Consistency evaluation (pass@k, pass^k metrics)
7. Trajectory optimality analysis
8. Baseline comparison framework

For research paper: "The AI Lab Manager: Autonomous Materials Discovery via Agentic Workflows"

References:
- Booking.com AI Agent Evaluation (2026): https://booking.ai/ai-agent-evaluation-82e781439d97
- τ-bench: A Benchmark for Tool-Agent-User Interaction (Yao et al., 2024)
- GAIA: A benchmark for General AI Assistants (Mialon et al., 2023)
"""

from .metrics import (
    ScreeningMetrics,
    InnovationMetrics,
    StabilityMetrics,
    AgentMetrics,
    compute_all_metrics,
)

from .benchmarks import (
    DiscoveryBenchmark,
    InnovationBenchmark,
    BenchmarkResult,
    run_benchmark_suite,
)

from .datasets import (
    create_discovery_benchmark_dataset,
    create_innovation_benchmark_dataset,
    load_ground_truth_data,
)

from .evaluator import (
    MaterialsResearchEvaluator,
    EvaluationReport,
)

# New: Tool Validation (Booking.com Glass Box Evaluation)
from .tool_validation import (
    ToolSchema,
    ToolCallError,
    ToolCallValidationResult,
    validate_tool_call,
    validate_tool_call_with_jsonschema,
    ToolCorrectnessResult,
    evaluate_tool_correctness,
    ToolReliabilityResult,
    check_tool_reliability,
    ToolValidationMetrics,
    compute_tool_validation_metrics,
)

# New: Consistency Evaluation (τ-bench methodology)
from .consistency import (
    ConsistencyResult,
    ConsistencyMetrics,
    QueryParaphraser,
    evaluate_consistency,
    Trajectory,
    TrajectoryOptimalityResult,
    TrajectoryMetrics,
    evaluate_trajectory_optimality,
    compute_trajectory_metrics,
    ComplexityAnalysis,
    analyze_complexity_scaling,
)

# New: Baseline Comparison (Booking.com decision framework)
from .baselines import (
    BaselineType,
    SystemResult,
    SystemEvaluation,
    ComparisonResult,
    CostEstimator,
    evaluate_system,
    compare_systems,
    create_zero_shot_baseline,
    create_few_shot_baseline,
    run_baseline_comparison,
)

__all__ = [
    # === CORE METRICS ===
    "ScreeningMetrics",
    "InnovationMetrics", 
    "StabilityMetrics",
    "AgentMetrics",
    "compute_all_metrics",
    
    # === BENCHMARKS ===
    "DiscoveryBenchmark",
    "InnovationBenchmark",
    "BenchmarkResult",
    "run_benchmark_suite",
    
    # === DATASETS ===
    "create_discovery_benchmark_dataset",
    "create_innovation_benchmark_dataset",
    "load_ground_truth_data",
    
    # === MAIN EVALUATOR ===
    "MaterialsResearchEvaluator",
    "EvaluationReport",
    
    # === TOOL VALIDATION (Glass Box) ===
    "ToolSchema",
    "ToolCallError",
    "ToolCallValidationResult",
    "validate_tool_call",
    "validate_tool_call_with_jsonschema",
    "ToolCorrectnessResult",
    "evaluate_tool_correctness",
    "ToolReliabilityResult",
    "check_tool_reliability",
    "ToolValidationMetrics",
    "compute_tool_validation_metrics",
    
    # === CONSISTENCY (τ-bench) ===
    "ConsistencyResult",
    "ConsistencyMetrics",
    "QueryParaphraser",
    "evaluate_consistency",
    "Trajectory",
    "TrajectoryOptimalityResult",
    "TrajectoryMetrics",
    "evaluate_trajectory_optimality",
    "compute_trajectory_metrics",
    "ComplexityAnalysis",
    "analyze_complexity_scaling",
    
    # === BASELINE COMPARISON ===
    "BaselineType",
    "SystemResult",
    "SystemEvaluation",
    "ComparisonResult",
    "CostEstimator",
    "evaluate_system",
    "compare_systems",
    "create_zero_shot_baseline",
    "create_few_shot_baseline",
    "run_baseline_comparison",
]
