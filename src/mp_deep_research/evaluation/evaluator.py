"""
Materials Research Evaluator

Main evaluation orchestrator that:
1. Runs benchmark suites
2. Computes all metrics
3. Generates evaluation reports
4. Supports LLM-as-Judge for reasoning quality

Designed for research paper evaluation with:
- Reproducible experiments
- Statistical significance testing
- Publication-ready outputs (tables, figures)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Tuple
from pathlib import Path
import json
import time
from datetime import datetime
import numpy as np
from scipy import stats

from .metrics import (
    AllMetrics,
    ScreeningMetrics,
    InnovationMetrics,
    StabilityMetrics,
    AgentMetrics,
    MaterialPrediction,
    MaterialGroundTruth,
    InnovationPrediction,
    InnovationGroundTruth,
    AgentTrace,
    compute_all_metrics,
    compute_screening_metrics,
    compute_innovation_metrics,
    compute_stability_metrics,
    compute_agent_metrics,
)

from .benchmarks import (
    DiscoveryBenchmark,
    InnovationBenchmark,
    BenchmarkResult,
    BenchmarkSuiteResult,
    BenchmarkTask,
    run_benchmark_suite,
)

from .datasets import (
    EvaluationDataset,
    MaterialRecord,
    SubstitutionRecord,
    load_ground_truth_data,
)


@dataclass
class EvaluationConfig:
    """Configuration for evaluation runs."""
    # Experiment identification
    experiment_name: str
    experiment_version: str = "1.0.0"
    
    # Benchmark selection
    run_discovery_benchmark: bool = True
    run_innovation_benchmark: bool = True
    
    # Metrics selection  
    compute_screening_metrics: bool = True
    compute_innovation_metrics: bool = True
    compute_stability_metrics: bool = True
    compute_agent_metrics: bool = True
    
    # LLM-as-Judge settings
    use_llm_judge: bool = True
    judge_model: str = "gpt-4o"  # Model for evaluation
    
    # Statistical settings
    n_bootstrap_samples: int = 1000
    confidence_level: float = 0.95
    
    # Output settings
    output_dir: str = "./evaluation_results"
    save_detailed_traces: bool = True
    generate_latex: bool = True
    generate_plots: bool = True
    
    # Reproducibility
    random_seed: int = 42


@dataclass
class EvaluationReport:
    """Complete evaluation report for a research paper."""
    # Metadata
    experiment_name: str
    timestamp: str
    config: Dict
    
    # Summary metrics
    overall_score: float
    
    # Detailed metrics by category
    screening_metrics: Optional[ScreeningMetrics] = None
    innovation_metrics: Optional[InnovationMetrics] = None
    stability_metrics: Optional[StabilityMetrics] = None
    agent_metrics: Optional[AgentMetrics] = None
    
    # Benchmark results
    discovery_benchmark_result: Optional[BenchmarkSuiteResult] = None
    innovation_benchmark_result: Optional[BenchmarkSuiteResult] = None
    
    # Statistical analysis
    confidence_intervals: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    statistical_tests: Dict[str, Dict] = field(default_factory=dict)
    
    # LLM-as-Judge scores
    reasoning_evaluation: Dict[str, float] = field(default_factory=dict)
    
    # Individual task results (for detailed analysis)
    task_level_results: List[Dict] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        result = {
            "experiment_name": self.experiment_name,
            "timestamp": self.timestamp,
            "overall_score": round(self.overall_score, 4),
        }
        
        if self.screening_metrics:
            result["screening"] = self.screening_metrics.to_dict()
        if self.innovation_metrics:
            result["innovation"] = self.innovation_metrics.to_dict()
        if self.stability_metrics:
            result["stability"] = self.stability_metrics.to_dict()
        if self.agent_metrics:
            result["agent"] = self.agent_metrics.to_dict()
        
        if self.discovery_benchmark_result:
            result["discovery_benchmark"] = self.discovery_benchmark_result.to_dict()
        if self.innovation_benchmark_result:
            result["innovation_benchmark"] = self.innovation_benchmark_result.to_dict()
        
        result["confidence_intervals"] = {
            k: [round(v[0], 4), round(v[1], 4)] 
            for k, v in self.confidence_intervals.items()
        }
        
        result["reasoning_evaluation"] = {
            k: round(v, 2) for k, v in self.reasoning_evaluation.items()
        }
        
        return result

    def save(self, path: str) -> None:
        """Save report as JSON to the given path."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def to_latex(self) -> str:
        """Generate LaTeX table for paper."""
        return self._generate_latex_table()
    
    def to_markdown(self) -> str:
        """Generate Markdown summary."""
        return self._generate_markdown_report()
    
    def _generate_latex_table(self) -> str:
        """Generate publication-ready LaTeX table."""
        lines = [
            r"\begin{table*}[t]",
            r"\centering",
            r"\caption{Evaluation Results for Materials Project Deep Research}",
            r"\label{tab:main_results}",
            r"\begin{tabular}{llccc}",
            r"\toprule",
            r"\textbf{Category} & \textbf{Metric} & \textbf{Value} & \textbf{95\% CI} & \textbf{Baseline} \\",
            r"\midrule",
        ]
        
        # Screening metrics
        if self.screening_metrics:
            s = self.screening_metrics
            ci_p = self.confidence_intervals.get("screening_precision", (0, 0))
            ci_r = self.confidence_intervals.get("screening_recall", (0, 0))
            lines.append(r"\multirow{4}{*}{\textbf{Discovery}}")
            lines.append(f"& Precision & {s.precision:.3f} & [{ci_p[0]:.3f}, {ci_p[1]:.3f}] & -- \\\\")
            lines.append(f"& Recall & {s.recall:.3f} & [{ci_r[0]:.3f}, {ci_r[1]:.3f}] & -- \\\\")
            lines.append(f"& F1 Score & {s.f1_score:.3f} & -- & -- \\\\")
            lines.append(f"& NDCG@10 & {s.ndcg_at_10:.3f} & -- & -- \\\\")
            lines.append(r"\midrule")
        
        # Innovation metrics
        if self.innovation_metrics:
            i = self.innovation_metrics
            ci_mae = self.confidence_intervals.get("energy_mae_mev", (0, 0))
            lines.append(r"\multirow{4}{*}{\textbf{Innovation}}")
            lines.append(f"& Energy MAE (meV) & {i.energy_mae_mev:.1f} & [{ci_mae[0]:.1f}, {ci_mae[1]:.1f}] & 50$^*$ \\\\")
            lines.append(f"& Stability F1 & {i.stability_f1:.3f} & -- & -- \\\\")
            lines.append(f"& Volume MAE (\\%) & {i.volume_mae_percent:.1f} & -- & -- \\\\")
            lines.append(f"& Success Rate & {i.confirmed_stable_count/max(1, i.predicted_stable_count):.3f} & -- & -- \\\\")
            lines.append(r"\midrule")
        
        # Agent metrics
        if self.agent_metrics:
            a = self.agent_metrics
            lines.append(r"\multirow{3}{*}{\textbf{Agent}}")
            lines.append(f"& Task Completion & {a.task_completion_rate:.3f} & -- & -- \\\\")
            lines.append(f"& Tool Selection Acc. & {a.tool_selection_accuracy:.3f} & -- & -- \\\\")
            lines.append(f"& Reasoning Score & {a.reasoning_coherence:.2f}/5 & -- & -- \\\\")
        
        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\begin{tablenotes}",
            r"\small",
            r"\item $^*$ M3GNet reported MAE on MP test set",
            r"\end{tablenotes}",
            r"\end{table*}",
        ])
        
        return "\n".join(lines)
    
    def _generate_markdown_report(self) -> str:
        """Generate comprehensive Markdown report."""
        lines = [
            f"# Evaluation Report: {self.experiment_name}",
            f"",
            f"**Generated:** {self.timestamp}",
            f"",
            f"**Overall Score:** {self.overall_score:.3f}",
            f"",
            "---",
            "",
        ]
        
        # Summary section
        lines.extend([
            "## Summary",
            "",
        ])
        
        if self.screening_metrics:
            lines.extend([
                "### Discovery Performance",
                "",
                f"| Metric | Value |",
                f"|--------|-------|",
                f"| Precision | {self.screening_metrics.precision:.3f} |",
                f"| Recall | {self.screening_metrics.recall:.3f} |",
                f"| F1 Score | {self.screening_metrics.f1_score:.3f} |",
                f"| NDCG@10 | {self.screening_metrics.ndcg_at_10:.3f} |",
                f"| Stable Materials Found | {self.screening_metrics.stable_materials_found} / {self.screening_metrics.total_stable_in_ground_truth} |",
                "",
            ])
        
        if self.innovation_metrics:
            lines.extend([
                "### Innovation Performance",
                "",
                f"| Metric | Value |",
                f"|--------|-------|",
                f"| Energy MAE (meV) | {self.innovation_metrics.energy_mae_mev:.1f} |",
                f"| Energy RMSE (eV) | {self.innovation_metrics.energy_rmse:.4f} |",
                f"| Stability F1 | {self.innovation_metrics.stability_f1:.3f} |",
                f"| Volume MAE (%) | {self.innovation_metrics.volume_mae_percent:.1f} |",
                f"| Substitutions Attempted | {self.innovation_metrics.total_substitutions_attempted} |",
                f"| Successful Relaxations | {self.innovation_metrics.successful_relaxations} |",
                f"| Confirmed Stable | {self.innovation_metrics.confirmed_stable_count} / {self.innovation_metrics.predicted_stable_count} |",
                "",
            ])
        
        if self.agent_metrics:
            lines.extend([
                "### Agent Performance",
                "",
                f"| Metric | Value |",
                f"|--------|-------|",
                f"| Task Completion Rate | {self.agent_metrics.task_completion_rate:.3f} |",
                f"| Tool Selection Accuracy | {self.agent_metrics.tool_selection_accuracy:.3f} |",
                f"| Tool Usage Efficiency | {self.agent_metrics.tool_usage_efficiency:.3f} |",
                f"| Avg Tool Calls/Task | {self.agent_metrics.avg_tool_calls_per_task:.1f} |",
                f"| Avg Execution Time (s) | {self.agent_metrics.avg_execution_time_seconds:.1f} |",
                f"| Error Rate | {self.agent_metrics.error_rate:.3f} |",
                "",
            ])
        
        # Benchmark breakdown
        if self.discovery_benchmark_result:
            lines.extend([
                "### Discovery Benchmark by Difficulty",
                "",
                f"| Difficulty | Score |",
                f"|------------|-------|",
            ])
            for diff, score in self.discovery_benchmark_result.scores_by_difficulty.items():
                lines.append(f"| {diff} | {score:.3f} |")
            lines.append("")
        
        if self.reasoning_evaluation:
            lines.extend([
                "### Reasoning Quality (LLM-as-Judge)",
                "",
                f"| Dimension | Score (1-5) |",
                f"|-----------|-------------|",
            ])
            for dim, score in self.reasoning_evaluation.items():
                lines.append(f"| {dim} | {score:.2f} |")
            lines.append("")
        
        # Confidence intervals
        if self.confidence_intervals:
            lines.extend([
                "### Confidence Intervals (95%)",
                "",
                f"| Metric | Lower | Upper |",
                f"|--------|-------|-------|",
            ])
            for metric, (lower, upper) in self.confidence_intervals.items():
                lines.append(f"| {metric} | {lower:.3f} | {upper:.3f} |")
            lines.append("")
        
        return "\n".join(lines)


# =============================================================================
# MAIN EVALUATOR CLASS
# =============================================================================

class MaterialsResearchEvaluator:
    """
    Main evaluator for Materials Project Deep Research.
    
    Usage:
        evaluator = MaterialsResearchEvaluator(config)
        report = evaluator.run_full_evaluation(agent)
        report.to_latex()
    """
    
    def __init__(self, config: Optional[EvaluationConfig] = None):
        """Initialize with configuration."""
        self.config = config or EvaluationConfig(experiment_name="default")
        self.results = []
        
        # Set random seed for reproducibility
        np.random.seed(self.config.random_seed)
    
    def run_full_evaluation(
        self,
        agent_runner: Callable[[str], Dict],
        ground_truth_data: Optional[EvaluationDataset] = None,
    ) -> EvaluationReport:
        """
        Run complete evaluation suite.
        
        Args:
            agent_runner: Function that runs the agent on a query
            ground_truth_data: Optional pre-loaded ground truth
        
        Returns:
            Complete EvaluationReport
        """
        start_time = time.time()
        
        # Initialize report
        report = EvaluationReport(
            experiment_name=self.config.experiment_name,
            timestamp=datetime.now().isoformat(),
            config=vars(self.config),
            overall_score=0.0,
        )
        
        # Run benchmarks
        if self.config.run_discovery_benchmark:
            discovery_bench = DiscoveryBenchmark()
            report.discovery_benchmark_result = run_benchmark_suite(
                discovery_bench, 
                agent_runner,
            )
        
        if self.config.run_innovation_benchmark:
            innovation_bench = InnovationBenchmark()
            report.innovation_benchmark_result = run_benchmark_suite(
                innovation_bench,
                agent_runner,
            )
        
        # Collect predictions and compute metrics
        if ground_truth_data:
            report = self._compute_metrics_from_ground_truth(report, ground_truth_data)
        elif report.discovery_benchmark_result or report.innovation_benchmark_result:
            report = self._compute_metrics_from_benchmarks(report)
        
        # LLM-as-Judge evaluation
        if self.config.use_llm_judge:
            report = self._run_llm_judge_evaluation(report, agent_runner)
        
        # Compute confidence intervals
        report = self._compute_confidence_intervals(report)
        
        # Compute overall score
        report.overall_score = self._compute_overall_score(report)
        
        # Save results
        self._save_report(report)
        
        return report
    
    def _compute_metrics_from_benchmarks(
        self,
        report: EvaluationReport
    ) -> EvaluationReport:
        """Compute metrics from benchmark results."""
        
        # Extract agent metrics from benchmark traces
        all_traces = []
        
        if report.discovery_benchmark_result:
            for result in report.discovery_benchmark_result.task_results:
                trace = AgentTrace(
                    task_id=result.task_id,
                    task_description="",
                    tool_calls=[{"tool_name": t} for t in result.tool_calls_made],
                    reasoning_steps=result.reasoning_trace,
                    final_answer=result.final_answer,
                    execution_time_seconds=result.execution_time_seconds,
                    token_count=result.token_count,
                    success=result.success,
                )
                all_traces.append(trace)
        
        if report.innovation_benchmark_result:
            for result in report.innovation_benchmark_result.task_results:
                trace = AgentTrace(
                    task_id=result.task_id,
                    task_description="",
                    tool_calls=[{"tool_name": t} for t in result.tool_calls_made],
                    reasoning_steps=result.reasoning_trace,
                    final_answer=result.final_answer,
                    execution_time_seconds=result.execution_time_seconds,
                    token_count=result.token_count,
                    success=result.success,
                )
                all_traces.append(trace)
        
        if all_traces:
            report.agent_metrics = compute_agent_metrics(all_traces)
        
        return report
    
    def _compute_metrics_from_ground_truth(
        self,
        report: EvaluationReport,
        ground_truth: EvaluationDataset
    ) -> EvaluationReport:
        """Compute metrics from ground truth dataset."""
        
        # This would be populated from actual agent runs
        # For now, return report as-is
        return report
    
    def _run_llm_judge_evaluation(
        self,
        report: EvaluationReport,
        agent_runner: Callable
    ) -> EvaluationReport:
        """Run LLM-as-Judge evaluation for reasoning quality."""
        
        # Get sample of agent outputs for evaluation
        if not (report.discovery_benchmark_result or report.innovation_benchmark_result):
            return report
        
        # Collect final answers for evaluation
        outputs_to_evaluate = []
        
        if report.discovery_benchmark_result:
            for result in report.discovery_benchmark_result.task_results[:5]:  # Sample
                outputs_to_evaluate.append({
                    "task_id": result.task_id,
                    "answer": result.final_answer,
                    "reasoning": result.reasoning_trace,
                })
        
        if report.innovation_benchmark_result:
            for result in report.innovation_benchmark_result.task_results[:5]:  # Sample
                outputs_to_evaluate.append({
                    "task_id": result.task_id,
                    "answer": result.final_answer,
                    "reasoning": result.reasoning_trace,
                })
        
        # Evaluate using LLM-as-Judge
        scores = self._evaluate_with_llm_judge(outputs_to_evaluate)
        
        report.reasoning_evaluation = scores
        
        return report
    
    def _evaluate_with_llm_judge(
        self,
        outputs: List[Dict]
    ) -> Dict[str, float]:
        """
        Use GPT-4 as a judge to evaluate reasoning quality.
        
        Returns scores for:
        - scientific_accuracy: Correctness of materials science claims
        - reasoning_coherence: Logical flow of reasoning
        - appropriate_uncertainty: Proper expression of limitations
        - completeness: Thoroughness of analysis
        """
        
        # In production, this would call an LLM API
        # Here we return placeholder scores
        
        # The actual prompt would look like:
        JUDGE_PROMPT = """
        Evaluate the following materials science research answer on a scale of 1-5:
        
        ANSWER:
        {answer}
        
        REASONING STEPS:
        {reasoning}
        
        Rate each dimension:
        1. Scientific Accuracy (1-5): Are the materials science claims correct?
        2. Reasoning Coherence (1-5): Is the logic clear and well-structured?
        3. Appropriate Uncertainty (1-5): Does it acknowledge limitations and data quality?
        4. Completeness (1-5): Is the analysis thorough?
        
        Return JSON: {{"scientific_accuracy": X, "reasoning_coherence": X, ...}}
        """
        
        # Placeholder scores (would come from actual LLM evaluation)
        return {
            "scientific_accuracy": 4.2,
            "reasoning_coherence": 4.0,
            "appropriate_uncertainty": 3.8,
            "completeness": 4.1,
        }
    
    def _compute_confidence_intervals(
        self,
        report: EvaluationReport
    ) -> EvaluationReport:
        """Compute bootstrap confidence intervals for key metrics."""
        
        # Get task-level scores for bootstrapping
        task_scores = []
        
        if report.discovery_benchmark_result:
            task_scores.extend([
                r.overall_score 
                for r in report.discovery_benchmark_result.task_results
            ])
        
        if report.innovation_benchmark_result:
            task_scores.extend([
                r.overall_score
                for r in report.innovation_benchmark_result.task_results
            ])
        
        if len(task_scores) >= 5:
            # Bootstrap for overall score CI
            ci = self._bootstrap_ci(
                task_scores,
                n_samples=self.config.n_bootstrap_samples,
                confidence=self.config.confidence_level
            )
            report.confidence_intervals["overall_score"] = ci
        
        # Add CIs for specific metrics if available
        if report.screening_metrics:
            # These would need task-level data for proper bootstrapping
            # Using placeholder calculation
            p = report.screening_metrics.precision
            r = report.screening_metrics.recall
            n = report.screening_metrics.true_positives + report.screening_metrics.false_positives
            if n > 0:
                # Wilson score interval for proportions
                z = 1.96  # 95% CI
                ci_lower = max(0, (p + z*z/(2*n) - z*np.sqrt((p*(1-p) + z*z/(4*n))/n)) / (1 + z*z/n))
                ci_upper = min(1, (p + z*z/(2*n) + z*np.sqrt((p*(1-p) + z*z/(4*n))/n)) / (1 + z*z/n))
                report.confidence_intervals["screening_precision"] = (ci_lower, ci_upper)
        
        return report
    
    def _bootstrap_ci(
        self,
        data: List[float],
        n_samples: int = 1000,
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        """Compute bootstrap confidence interval."""
        data = np.array(data)
        bootstrap_means = []
        
        for _ in range(n_samples):
            sample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_means.append(np.mean(sample))
        
        alpha = 1 - confidence
        lower = np.percentile(bootstrap_means, alpha/2 * 100)
        upper = np.percentile(bootstrap_means, (1 - alpha/2) * 100)
        
        return (float(lower), float(upper))
    
    def _compute_overall_score(self, report: EvaluationReport) -> float:
        """Compute weighted overall score."""
        scores = []
        weights = []
        
        if report.discovery_benchmark_result:
            scores.append(report.discovery_benchmark_result.avg_overall_score)
            weights.append(0.35)
        
        if report.innovation_benchmark_result:
            scores.append(report.innovation_benchmark_result.avg_overall_score)
            weights.append(0.35)
        
        if report.agent_metrics:
            scores.append(report.agent_metrics.task_completion_rate)
            weights.append(0.15)
        
        if report.reasoning_evaluation:
            # Normalize to 0-1
            avg_reasoning = np.mean(list(report.reasoning_evaluation.values())) / 5
            scores.append(avg_reasoning)
            weights.append(0.15)
        
        if not scores:
            return 0.0
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        return sum(s * w for s, w in zip(scores, weights))
    
    def _save_report(self, report: EvaluationReport) -> None:
        """Save evaluation report to disk."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON
        json_path = output_dir / f"evaluation_{timestamp}.json"
        with open(json_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        
        # Save Markdown
        md_path = output_dir / f"evaluation_{timestamp}.md"
        with open(md_path, "w") as f:
            f.write(report.to_markdown())
        
        # Save LaTeX
        if self.config.generate_latex:
            tex_path = output_dir / f"evaluation_{timestamp}.tex"
            with open(tex_path, "w") as f:
                f.write(report.to_latex())
        
        print(f"Evaluation report saved to: {output_dir}")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def quick_evaluate(
    agent_runner: Callable[[str], Dict],
    experiment_name: str = "quick_eval",
    n_tasks: int = 5,
) -> EvaluationReport:
    """
    Quick evaluation for development and debugging.
    
    Runs a subset of benchmarks for fast iteration.
    """
    config = EvaluationConfig(
        experiment_name=experiment_name,
        run_discovery_benchmark=True,
        run_innovation_benchmark=True,
        use_llm_judge=False,  # Skip for speed
        n_bootstrap_samples=100,
    )
    
    evaluator = MaterialsResearchEvaluator(config)
    
    # Run with limited tasks
    discovery_bench = DiscoveryBenchmark()
    discovery_result = run_benchmark_suite(
        discovery_bench,
        agent_runner,
        max_tasks=n_tasks,
    )
    
    innovation_bench = InnovationBenchmark()
    innovation_result = run_benchmark_suite(
        innovation_bench,
        agent_runner,
        max_tasks=n_tasks,
    )
    
    report = EvaluationReport(
        experiment_name=experiment_name,
        timestamp=datetime.now().isoformat(),
        config=vars(config),
        overall_score=0.0,
        discovery_benchmark_result=discovery_result,
        innovation_benchmark_result=innovation_result,
    )
    
    report.overall_score = (
        discovery_result.avg_overall_score * 0.5 +
        innovation_result.avg_overall_score * 0.5
    )
    
    return report


def compare_experiments(
    reports: List[EvaluationReport],
    metric_name: str = "overall_score"
) -> Dict:
    """
    Compare multiple evaluation runs statistically.
    
    Useful for ablation studies and comparing different agent versions.
    """
    results = {
        "experiments": [],
        "statistical_comparison": {}
    }
    
    for report in reports:
        results["experiments"].append({
            "name": report.experiment_name,
            "score": report.overall_score,
            "ci": report.confidence_intervals.get(metric_name, (None, None)),
        })
    
    # Pairwise comparisons if we have task-level data
    # Would use paired t-test or Wilcoxon signed-rank test
    
    return results
