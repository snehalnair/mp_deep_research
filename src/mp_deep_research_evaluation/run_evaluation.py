#!/usr/bin/env python3
"""
Example: Running the Full Evaluation Suite

This script demonstrates how to:
1. Set up the evaluation framework
2. Run discovery and innovation benchmarks
3. Compute all metrics
4. Generate publication-ready reports

For your research paper, you can customize the benchmarks and
add domain-specific evaluation criteria.
"""

import os
import sys
from pathlib import Path

# Add src to path if running from project root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mp_deep_research.evaluation import (
    # Evaluator
    MaterialsResearchEvaluator,
    EvaluationConfig,
    EvaluationReport,
    quick_evaluate,
    
    # Benchmarks
    DiscoveryBenchmark,
    InnovationBenchmark,
    run_benchmark_suite,
    
    # Datasets
    create_discovery_benchmark_dataset,
    create_innovation_benchmark_dataset,
    load_ground_truth_data,
    
    # Metrics
    compute_all_metrics,
    AllMetrics,
)


def create_mock_agent():
    """
    Create a mock agent for testing the evaluation framework.
    
    In production, replace this with your actual agent.
    """
    def mock_agent_runner(query: str) -> dict:
        """
        Mock agent that simulates reasonable responses.
        
        Replace this with your actual LangGraph agent invocation:
        
            from mp_deep_research.research_agent import research_agent
            result = research_agent.invoke({"messages": [{"role": "user", "content": query}]})
            return parse_agent_output(result)
        """
        # Simple pattern matching for demonstration
        response = {
            "final_answer": "",
            "tool_calls": [],
            "reasoning": [],
            "materials_mentioned": [],
            "properties": {},
            "token_count": 500,
        }
        
        query_lower = query.lower()
        
        # Discovery queries
        if "li-co-o" in query_lower or "licoo2" in query_lower:
            response["final_answer"] = """
            Found stable lithium cobalt oxide materials in the Li-Co-O system.
            Top candidate: LiCoO2 (mp-22526) with E_above_hull = 0 meV/atom.
            This is the classic layered cathode material used in Li-ion batteries.
            """
            response["tool_calls"] = [
                {"name": "batch_screen_materials", "args": {"chemsys": "Li-Co-O"}},
            ]
            response["materials_mentioned"] = ["mp-22526"]
            response["properties"] = {"chemical_system": "Li-Co-O", "stability_threshold": 0.025}
            response["reasoning"] = [
                "User wants Li-ion cathode materials",
                "Searching Li-Co-O system",
                "Found LiCoO2 as most stable",
            ]
        
        elif "li-fe-p-o" in query_lower or "lifepo4" in query_lower:
            response["final_answer"] = """
            Found LiFePO4 (mp-19017) as the most stable olivine cathode.
            E_above_hull = 0 meV/atom, experimentally validated (ICSD).
            """
            response["tool_calls"] = [
                {"name": "batch_screen_materials", "args": {"chemsys": "Li-Fe-P-O"}},
                {"name": "analyze_candidate", "args": {"material_id": "mp-19017"}},
            ]
            response["materials_mentioned"] = ["mp-19017"]
            response["properties"] = {"chemical_system": "Li-Fe-P-O"}
            
        elif "substitute" in query_lower or "replace" in query_lower:
            response["final_answer"] = """
            Performed substitution: Li → Na in LiFePO4 (mp-19017)
            New composition: NaFePO4
            M3GNet relaxation completed.
            E_above_hull: ~45 meV/atom (metastable)
            Recommendation: May be synthesizable but less stable than parent.
            """
            response["tool_calls"] = [
                {"name": "substitute_species", "args": {"material_id": "mp-19017", "substitutions": {"Li": "Na"}}},
                {"name": "relax_structure_m3gnet", "args": {"structure_path": "output.json"}},
                {"name": "assess_stability", "args": {"composition": "NaFePO4"}},
            ]
            response["materials_mentioned"] = ["mp-19017"]
            
        elif "compare" in query_lower:
            response["final_answer"] = """
            Comparison of olivine cathodes:
            - LiFePO4 (mp-19017): E_hull=0 meV, highly stable, experimental
            - LiMnPO4 (mp-18767): E_hull=0 meV, stable, higher voltage
            
            LiFePO4 recommended for stability, LiMnPO4 for voltage.
            """
            response["tool_calls"] = [
                {"name": "batch_screen_materials", "args": {"chemsys": "Li-Fe-Mn-P-O"}},
                {"name": "analyze_candidate", "args": {"material_id": "mp-19017"}},
                {"name": "analyze_candidate", "args": {"material_id": "mp-18767"}},
            ]
            response["materials_mentioned"] = ["mp-19017", "mp-18767"]
            response["reasoning"] = [
                "User wants comparison",
                "Analyzed both materials",
                "LiFePO4 more stable, LiMnPO4 higher voltage",
            ]
            
        else:
            # Generic response
            response["final_answer"] = f"Processed query: {query[:50]}..."
            response["tool_calls"] = [{"name": "batch_screen_materials", "args": {}}]
        
        return response
    
    return mock_agent_runner


def example_quick_evaluation():
    """
    Example: Quick evaluation for development.
    
    Use this during development to quickly check your agent.
    """
    print("=" * 60)
    print("QUICK EVALUATION (Development Mode)")
    print("=" * 60)
    
    agent = create_mock_agent()
    
    report = quick_evaluate(
        agent_runner=agent,
        experiment_name="quick_dev_test",
        n_tasks=3,
    )
    
    print(f"\nOverall Score: {report.overall_score:.3f}")
    
    if report.discovery_benchmark_result:
        print(f"\nDiscovery Benchmark:")
        print(f"  Tasks Completed: {report.discovery_benchmark_result.completed_tasks}/{report.discovery_benchmark_result.total_tasks}")
        print(f"  Avg Score: {report.discovery_benchmark_result.avg_overall_score:.3f}")
    
    if report.innovation_benchmark_result:
        print(f"\nInnovation Benchmark:")
        print(f"  Tasks Completed: {report.innovation_benchmark_result.completed_tasks}/{report.innovation_benchmark_result.total_tasks}")
        print(f"  Avg Score: {report.innovation_benchmark_result.avg_overall_score:.3f}")
    
    return report


def example_full_evaluation():
    """
    Example: Full evaluation for paper submission.
    
    This runs the complete evaluation suite with all metrics.
    """
    print("=" * 60)
    print("FULL EVALUATION (Paper Submission Mode)")
    print("=" * 60)
    
    # Configure evaluation
    config = EvaluationConfig(
        experiment_name="MPDeepResearch_v1",
        experiment_version="1.0.0",
        
        # Run all benchmarks
        run_discovery_benchmark=True,
        run_innovation_benchmark=True,
        
        # Compute all metrics
        compute_screening_metrics=True,
        compute_innovation_metrics=True,
        compute_stability_metrics=True,
        compute_agent_metrics=True,
        
        # LLM-as-Judge (disable for testing)
        use_llm_judge=False,  # Set True for full evaluation
        judge_model="gpt-4o",
        
        # Statistical settings
        n_bootstrap_samples=1000,
        confidence_level=0.95,
        
        # Output
        output_dir="./evaluation_results",
        generate_latex=True,
        generate_plots=True,
        
        # Reproducibility
        random_seed=42,
    )
    
    # Create evaluator
    evaluator = MaterialsResearchEvaluator(config)
    
    # Create agent
    agent = create_mock_agent()
    
    # Run evaluation
    print("\nRunning benchmarks...")
    report = evaluator.run_full_evaluation(agent_runner=agent)
    
    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    print(f"\nOverall Score: {report.overall_score:.3f}")
    
    if report.confidence_intervals:
        ci = report.confidence_intervals.get("overall_score", (0, 0))
        print(f"95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]")
    
    # Print detailed metrics
    if report.screening_metrics:
        print("\n--- Discovery Metrics ---")
        print(f"  Precision: {report.screening_metrics.precision:.3f}")
        print(f"  Recall: {report.screening_metrics.recall:.3f}")
        print(f"  F1: {report.screening_metrics.f1_score:.3f}")
        print(f"  NDCG@10: {report.screening_metrics.ndcg_at_10:.3f}")
    
    if report.innovation_metrics:
        print("\n--- Innovation Metrics ---")
        print(f"  Energy MAE: {report.innovation_metrics.energy_mae_mev:.1f} meV")
        print(f"  Stability F1: {report.innovation_metrics.stability_f1:.3f}")
    
    if report.agent_metrics:
        print("\n--- Agent Metrics ---")
        print(f"  Task Completion: {report.agent_metrics.task_completion_rate:.3f}")
        print(f"  Tool Selection Accuracy: {report.agent_metrics.tool_selection_accuracy:.3f}")
    
    if report.reasoning_evaluation:
        print("\n--- Reasoning Quality (LLM-as-Judge) ---")
        for dim, score in report.reasoning_evaluation.items():
            print(f"  {dim}: {score:.2f}/5")
    
    # Generate outputs
    print("\n--- Generated Outputs ---")
    print(f"  JSON report saved")
    print(f"  Markdown report saved")
    if config.generate_latex:
        print(f"  LaTeX table saved")
    
    # Print LaTeX table for paper
    print("\n--- LaTeX Table (for paper) ---")
    print(report.to_latex()[:500] + "...")
    
    return report


def example_benchmark_inspection():
    """
    Example: Inspect benchmark tasks.
    
    Useful for understanding what the benchmarks test.
    """
    print("=" * 60)
    print("BENCHMARK INSPECTION")
    print("=" * 60)
    
    # Discovery benchmark
    discovery = DiscoveryBenchmark()
    print(f"\nDiscovery Benchmark: {len(discovery)} tasks")
    
    for diff in ["easy", "medium", "hard"]:
        from mp_deep_research.evaluation.benchmarks import TaskDifficulty
        tasks = discovery.get_tasks_by_difficulty(TaskDifficulty(diff))
        print(f"  {diff.upper()}: {len(tasks)} tasks")
    
    print("\nSample Discovery Tasks:")
    for task in discovery.tasks[:3]:
        print(f"\n  [{task.task_id}] ({task.difficulty.value})")
        print(f"  Query: {task.user_query[:80]}...")
        print(f"  Expected: {task.expected_tool_sequence}")
    
    # Innovation benchmark
    innovation = InnovationBenchmark()
    print(f"\n\nInnovation Benchmark: {len(innovation)} tasks")
    
    print("\nSample Innovation Tasks:")
    for task in innovation.tasks[:3]:
        print(f"\n  [{task.task_id}] ({task.difficulty.value})")
        print(f"  Query: {task.user_query[:80]}...")
    
    # Ground truth data
    print("\n\nGround Truth Data Available:")
    for formula, data in innovation.ground_truth_energies.items():
        print(f"  {formula}: E_hull = {data['dft_e_above_hull_mev']} meV")


def example_dataset_generation():
    """
    Example: Generate and inspect evaluation datasets.
    """
    print("=" * 60)
    print("DATASET GENERATION")
    print("=" * 60)
    
    # Create discovery dataset (without API)
    print("\nCreating Discovery Dataset (synthetic)...")
    discovery_data = create_discovery_benchmark_dataset(use_api=False)
    
    print(f"  Materials: {len(discovery_data.materials)}")
    print(f"  Statistics: {discovery_data.statistics}")
    
    print("\n  Sample Materials:")
    for mat in discovery_data.materials[:3]:
        print(f"    {mat.material_id}: {mat.formula} (E_hull={mat.e_above_hull:.0f} meV)")
    
    # Create innovation dataset
    print("\nCreating Innovation Dataset...")
    innovation_data = create_innovation_benchmark_dataset()
    
    print(f"  Substitutions: {len(innovation_data.substitutions)}")
    
    print("\n  Sample Substitutions:")
    for sub in innovation_data.substitutions[:3]:
        print(f"    {sub.parent_formula} + {sub.substitutions} → {sub.child_formula}")
        if sub.dft_e_above_hull is not None:
            print(f"      DFT E_hull: {sub.dft_e_above_hull} meV")


def example_custom_benchmark():
    """
    Example: Create a custom benchmark for your specific needs.
    """
    print("=" * 60)
    print("CUSTOM BENCHMARK CREATION")
    print("=" * 60)
    
    from mp_deep_research.evaluation.benchmarks import BenchmarkTask, TaskType, TaskDifficulty
    
    # Create custom task for your specific research
    custom_tasks = [
        BenchmarkTask(
            task_id="custom_001",
            task_type=TaskType.DISCOVERY,
            difficulty=TaskDifficulty.MEDIUM,
            description="Find sodium-ion battery cathode materials",
            user_query="Search for stable sodium cathode materials in the Na-Fe-P-O and Na-Mn-O systems. Compare stability and expected voltage.",
            expected_materials=["mp-12345"],  # Your expected answers
            expected_tool_sequence=["batch_screen_materials", "batch_screen_materials", "analyze_candidate"],
            success_criteria="Identifies Na analogues of LiFePO4",
            target_application="Na-ion battery",
        ),
        BenchmarkTask(
            task_id="custom_002",
            task_type=TaskType.INNOVATION,
            difficulty=TaskDifficulty.HARD,
            description="Design Co-free NMC cathode",
            user_query="Starting from NMC811 (LiNi0.8Mn0.1Co0.1O2), design a cobalt-free variant. Test Ni-rich compositions with Al or Ti doping.",
            expected_materials=[],
            expected_tool_sequence=["run_innovation_flow", "run_innovation_flow", "think"],
            success_criteria="Proposes viable Co-free alternatives with stability analysis",
            target_application="Sustainable Li-ion cathode",
        ),
    ]
    
    print(f"Created {len(custom_tasks)} custom benchmark tasks")
    
    for task in custom_tasks:
        print(f"\n  Task: {task.task_id}")
        print(f"  Description: {task.description}")
        print(f"  Application: {task.target_application}")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("MATERIALS PROJECT DEEP RESEARCH - EVALUATION EXAMPLES")
    print("=" * 60)
    
    # Run examples
    print("\n\n[1/5] Benchmark Inspection")
    example_benchmark_inspection()
    
    print("\n\n[2/5] Dataset Generation")
    example_dataset_generation()
    
    print("\n\n[3/5] Quick Evaluation")
    quick_report = example_quick_evaluation()
    
    print("\n\n[4/5] Full Evaluation")
    full_report = example_full_evaluation()
    
    print("\n\n[5/5] Custom Benchmark")
    example_custom_benchmark()
    
    print("\n\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print("\nNext steps for your paper:")
    print("1. Replace mock agent with your actual LangGraph agent")
    print("2. Run full evaluation with LLM-as-Judge enabled")
    print("3. Add custom benchmarks for your specific use cases")
    print("4. Use generated LaTeX tables in your paper")
    print("5. Include confidence intervals for statistical rigor")


if __name__ == "__main__":
    main()
