"""
Tests for the Evaluation Framework

Tests for:
- Metrics computation
- Benchmark task creation and execution
- Dataset generation
- Full evaluation pipeline
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
import json
from pathlib import Path

# Import evaluation modules
from mp_deep_research.evaluation.metrics import (
    ScreeningMetrics,
    InnovationMetrics,
    StabilityMetrics,
    AgentMetrics,
    MaterialPrediction,
    MaterialGroundTruth,
    InnovationPrediction,
    InnovationGroundTruth,
    AgentTrace,
    compute_screening_metrics,
    compute_innovation_metrics,
    compute_stability_metrics,
    compute_agent_metrics,
    AllMetrics,
)

from mp_deep_research.evaluation.benchmarks import (
    DiscoveryBenchmark,
    InnovationBenchmark,
    BenchmarkTask,
    TaskDifficulty,
    TaskType,
    run_benchmark_suite,
    BenchmarkResult,
)

from mp_deep_research.evaluation.datasets import (
    create_discovery_benchmark_dataset,
    create_innovation_benchmark_dataset,
    MaterialRecord,
    SubstitutionRecord,
    EvaluationDataset,
)


# =============================================================================
# METRICS TESTS
# =============================================================================

class TestScreeningMetrics:
    """Tests for screening/discovery metrics."""
    
    @pytest.fixture
    def sample_predictions(self):
        """Create sample predictions."""
        return [
            MaterialPrediction(
                material_id="mp-22526",
                formula="LiCoO2",
                predicted_stable=True,
                predicted_e_hull=0,
                rank_position=1,
            ),
            MaterialPrediction(
                material_id="mp-19017",
                formula="LiFePO4",
                predicted_stable=True,
                predicted_e_hull=0,
                rank_position=2,
            ),
            MaterialPrediction(
                material_id="mp-12345",
                formula="LiFeO2",
                predicted_stable=True,
                predicted_e_hull=30,  # Actually unstable
                rank_position=3,
            ),
        ]
    
    @pytest.fixture
    def sample_ground_truth(self):
        """Create sample ground truth."""
        return [
            MaterialGroundTruth(
                material_id="mp-22526",
                formula="LiCoO2",
                is_stable=True,
                e_above_hull=0,
            ),
            MaterialGroundTruth(
                material_id="mp-19017",
                formula="LiFePO4",
                is_stable=True,
                e_above_hull=0,
            ),
            MaterialGroundTruth(
                material_id="mp-12345",
                formula="LiFeO2",
                is_stable=False,
                e_above_hull=50,  # Unstable
            ),
            MaterialGroundTruth(
                material_id="mp-99999",
                formula="LiNiO2",
                is_stable=True,
                e_above_hull=10,  # This one was missed
            ),
        ]
    
    def test_compute_screening_metrics(self, sample_predictions, sample_ground_truth):
        """Test basic screening metrics computation."""
        metrics = compute_screening_metrics(
            sample_predictions,
            sample_ground_truth,
            stability_threshold_mev=25.0
        )
        
        # Check classification
        assert metrics.true_positives == 2  # LiCoO2, LiFePO4
        assert metrics.false_positives == 1  # LiFeO2 (predicted stable, actually not)
        assert metrics.false_negatives == 1  # LiNiO2 (missed)
        
        # Check precision/recall
        # Precision = TP / (TP + FP) = 2/3
        assert abs(metrics.precision - 2/3) < 0.01
        
        # Recall = TP / (TP + FN) = 2/3
        assert abs(metrics.recall - 2/3) < 0.01
        
        # F1
        expected_f1 = 2 * (2/3) * (2/3) / (2/3 + 2/3)
        assert abs(metrics.f1_score - expected_f1) < 0.01
    
    def test_metrics_to_dict(self, sample_predictions, sample_ground_truth):
        """Test metrics serialization."""
        metrics = compute_screening_metrics(sample_predictions, sample_ground_truth)
        d = metrics.to_dict()
        
        assert "precision" in d
        assert "recall" in d
        assert "f1_score" in d
        assert "ndcg@10" in d
        assert isinstance(d["precision"], float)
    
    def test_perfect_predictions(self):
        """Test metrics with perfect predictions."""
        predictions = [
            MaterialPrediction("mp-1", "A", True, 0, 1),
            MaterialPrediction("mp-2", "B", True, 0, 2),
        ]
        ground_truth = [
            MaterialGroundTruth("mp-1", "A", True, 0),
            MaterialGroundTruth("mp-2", "B", True, 0),
        ]
        
        metrics = compute_screening_metrics(predictions, ground_truth)
        
        assert metrics.precision == 1.0
        assert metrics.recall == 1.0
        assert metrics.f1_score == 1.0


class TestInnovationMetrics:
    """Tests for innovation/generative metrics."""
    
    @pytest.fixture
    def sample_predictions(self):
        """Create sample innovation predictions."""
        return [
            InnovationPrediction(
                parent_id="mp-19017",
                substitutions={"Li": "Na"},
                new_formula="NaFePO4",
                predicted_energy_per_atom=-6.8,
                predicted_e_above_hull=40,  # Predicts metastable
                predicted_stable=False,
                volume_change_percent=12.0,
                relaxation_converged=True,
            ),
            InnovationPrediction(
                parent_id="mp-19017",
                substitutions={"Fe": "Mn"},
                new_formula="LiMnPO4",
                predicted_energy_per_atom=-6.75,
                predicted_e_above_hull=5,  # Predicts stable
                predicted_stable=True,
                volume_change_percent=2.5,
                relaxation_converged=True,
            ),
        ]
    
    @pytest.fixture
    def sample_ground_truth(self):
        """Create sample DFT ground truth."""
        return [
            InnovationGroundTruth(
                parent_id="mp-19017",
                substitutions={"Li": "Na"},
                new_formula="NaFePO4",
                dft_energy_per_atom=-6.832,
                dft_e_above_hull=45,  # Actually metastable
                is_stable=False,
                dft_volume_change_percent=12.5,
            ),
            InnovationGroundTruth(
                parent_id="mp-19017",
                substitutions={"Fe": "Mn"},
                new_formula="LiMnPO4",
                dft_energy_per_atom=-6.754,
                dft_e_above_hull=0,  # Actually stable
                is_stable=True,
                dft_volume_change_percent=2.3,
            ),
        ]
    
    def test_compute_innovation_metrics(self, sample_predictions, sample_ground_truth):
        """Test innovation metrics computation."""
        metrics = compute_innovation_metrics(
            sample_predictions,
            sample_ground_truth,
            stability_threshold_mev=25.0
        )
        
        # Check counts
        assert metrics.total_substitutions_attempted == 2
        assert metrics.successful_relaxations == 2
        
        # Check energy error (average of |0.032| and |0.004|)
        # Error for NaFePO4: -6.8 - (-6.832) = 0.032
        # Error for LiMnPO4: -6.75 - (-6.754) = 0.004
        expected_mae = (0.032 + 0.004) / 2
        assert abs(metrics.energy_mae - expected_mae) < 0.01
        
        # Check stability accuracy
        # NaFePO4: predicted not stable, actually not stable → correct
        # LiMnPO4: predicted stable, actually stable → correct
        assert metrics.stability_accuracy == 1.0
    
    def test_innovation_metrics_to_dict(self, sample_predictions, sample_ground_truth):
        """Test innovation metrics serialization."""
        metrics = compute_innovation_metrics(sample_predictions, sample_ground_truth)
        d = metrics.to_dict()
        
        assert "energy_mae_ev" in d
        assert "energy_mae_mev" in d
        assert "stability_f1" in d


class TestStabilityMetrics:
    """Tests for stability assessment metrics."""
    
    def test_compute_stability_metrics(self):
        """Test stability metrics computation."""
        predicted = [0, 20, 50, 100, 150]  # meV
        actual = [5, 25, 45, 90, 140]  # meV
        
        metrics = compute_stability_metrics(predicted, actual)
        
        # Check MAE
        errors = [abs(p - a) for p, a in zip(predicted, actual)]
        expected_mae = np.mean(errors)
        assert abs(metrics.e_hull_mae_mev - expected_mae) < 0.1
        
        # Check classification accuracy
        # Classification: <25 stable, 25-50 metastable, 50-100 marginal, >=100 unstable
        # Predicted: stable, stable, marginal, marginal, unstable
        # Actual: stable, metastable, metastable, marginal, unstable
        # Correct: stable (yes), stable vs metastable (no), marginal vs metastable (no), 
        #          marginal (yes), unstable (yes)
        # 3/5 = 0.6
        assert abs(metrics.verdict_accuracy - 0.6) < 0.01


class TestAgentMetrics:
    """Tests for agent-level metrics."""
    
    @pytest.fixture
    def sample_traces(self):
        """Create sample agent traces."""
        return [
            AgentTrace(
                task_id="task_1",
                task_description="Find cathode materials",
                tool_calls=[
                    {"tool_name": "batch_screen_materials"},
                    {"tool_name": "analyze_candidate"},
                ],
                reasoning_steps=["Searched database", "Analyzed top candidate"],
                final_answer="Found LiCoO2 as best candidate",
                execution_time_seconds=5.0,
                token_count=1000,
                success=True,
            ),
            AgentTrace(
                task_id="task_2",
                task_description="Design new material",
                tool_calls=[
                    {"tool_name": "substitute_species"},
                    {"tool_name": "relax_structure_m3gnet"},
                ],
                reasoning_steps=["Made substitution", "Relaxed structure"],
                final_answer="Created NaFePO4",
                execution_time_seconds=10.0,
                token_count=2000,
                success=True,
            ),
            AgentTrace(
                task_id="task_3",
                task_description="Failed task",
                tool_calls=[{"tool_name": "batch_screen_materials"}],
                reasoning_steps=[],
                final_answer="Error occurred",
                execution_time_seconds=2.0,
                token_count=500,
                success=False,
            ),
        ]
    
    def test_compute_agent_metrics(self, sample_traces):
        """Test agent metrics computation."""
        metrics = compute_agent_metrics(sample_traces)
        
        # Task completion
        assert abs(metrics.task_completion_rate - 2/3) < 0.01
        
        # Average tool calls
        expected_avg = (2 + 2 + 1) / 3
        assert abs(metrics.avg_tool_calls_per_task - expected_avg) < 0.01
        
        # Execution time
        expected_time = (5 + 10 + 2) / 3
        assert abs(metrics.avg_execution_time_seconds - expected_time) < 0.01
    
    def test_agent_metrics_with_expected_tools(self, sample_traces):
        """Test agent metrics with expected tool sequences."""
        expected_sequences = {
            "task_1": ["batch_screen_materials", "analyze_candidate"],
            "task_2": ["substitute_species", "relax_structure_m3gnet"],
        }
        
        metrics = compute_agent_metrics(sample_traces, expected_sequences)
        
        # Both task_1 and task_2 used expected tools
        assert metrics.tool_selection_accuracy == 1.0


# =============================================================================
# BENCHMARK TESTS
# =============================================================================

class TestDiscoveryBenchmark:
    """Tests for discovery benchmark."""
    
    def test_benchmark_creation(self):
        """Test benchmark task creation."""
        benchmark = DiscoveryBenchmark()
        
        assert len(benchmark) > 0
        assert len(benchmark.tasks) > 0
    
    def test_task_structure(self):
        """Test that tasks have required fields."""
        benchmark = DiscoveryBenchmark()
        
        for task in benchmark.tasks:
            assert task.task_id
            assert task.user_query
            assert task.task_type == TaskType.DISCOVERY
            assert task.difficulty in TaskDifficulty
            assert task.expected_tool_sequence
    
    def test_difficulty_distribution(self):
        """Test that benchmark has tasks of all difficulties."""
        benchmark = DiscoveryBenchmark()
        
        easy = benchmark.get_tasks_by_difficulty(TaskDifficulty.EASY)
        medium = benchmark.get_tasks_by_difficulty(TaskDifficulty.MEDIUM)
        hard = benchmark.get_tasks_by_difficulty(TaskDifficulty.HARD)
        
        assert len(easy) >= 2
        assert len(medium) >= 2
        assert len(hard) >= 2


class TestInnovationBenchmark:
    """Tests for innovation benchmark."""
    
    def test_benchmark_creation(self):
        """Test benchmark task creation."""
        benchmark = InnovationBenchmark()
        
        assert len(benchmark) > 0
        assert len(benchmark.tasks) > 0
    
    def test_ground_truth_available(self):
        """Test that ground truth data is available."""
        benchmark = InnovationBenchmark()
        
        # Check some known formulas
        gt = benchmark.get_ground_truth_for_formula("NaFePO4")
        if gt:  # May not have all ground truth
            assert "dft_e_above_hull_mev" in gt


class TestBenchmarkRunner:
    """Tests for benchmark execution."""
    
    def test_run_benchmark_suite(self):
        """Test running a benchmark suite."""
        benchmark = DiscoveryBenchmark()
        
        # Mock agent runner
        def mock_agent(query):
            return {
                "final_answer": "Found LiCoO2 mp-22526",
                "tool_calls": [{"name": "batch_screen_materials"}],
                "reasoning": ["Searched database"],
                "materials_mentioned": ["mp-22526"],
                "properties": {"chemical_system": "Li-Co-O"},
            }
        
        result = run_benchmark_suite(
            benchmark,
            mock_agent,
            max_tasks=2,  # Limit for testing
        )
        
        assert result.total_tasks == 2
        assert result.completed_tasks <= 2
        assert len(result.task_results) == 2


# =============================================================================
# DATASET TESTS
# =============================================================================

class TestDatasets:
    """Tests for dataset generation."""
    
    def test_create_discovery_dataset(self):
        """Test discovery dataset creation (without API)."""
        dataset = create_discovery_benchmark_dataset(
            use_api=False,  # Use synthetic data
        )
        
        assert dataset.name
        assert len(dataset.materials) > 0
        
        # Check material records
        for mat in dataset.materials:
            assert mat.material_id
            assert mat.formula
            assert mat.e_above_hull >= 0
    
    def test_create_innovation_dataset(self):
        """Test innovation dataset creation."""
        dataset = create_innovation_benchmark_dataset()
        
        assert len(dataset.substitutions) > 0
        
        # Check substitution records
        for sub in dataset.substitutions:
            assert sub.parent_id
            assert sub.substitutions
            assert sub.child_formula
    
    def test_dataset_statistics(self):
        """Test dataset statistics computation."""
        dataset = create_discovery_benchmark_dataset(use_api=False)
        
        assert "n_materials" in dataset.statistics
        assert "n_stable" in dataset.statistics


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestEvaluationPipeline:
    """Integration tests for full evaluation pipeline."""
    
    def test_all_metrics_computation(self):
        """Test computing all metrics together."""
        # Create sample data
        screening_preds = [
            MaterialPrediction("mp-1", "A", True, 0, 1),
            MaterialPrediction("mp-2", "B", True, 10, 2),
        ]
        screening_gt = [
            MaterialGroundTruth("mp-1", "A", True, 0),
            MaterialGroundTruth("mp-2", "B", True, 5),
        ]
        
        all_metrics = AllMetrics()
        all_metrics.screening = compute_screening_metrics(screening_preds, screening_gt)
        
        # Check we can serialize
        d = all_metrics.to_dict()
        assert "screening" in d
    
    def test_latex_generation(self):
        """Test LaTeX table generation."""
        all_metrics = AllMetrics()
        all_metrics.screening = ScreeningMetrics(
            precision=0.85,
            recall=0.90,
            f1_score=0.87,
            ndcg_at_10=0.82,
        )
        all_metrics.innovation = InnovationMetrics(
            energy_mae_mev=35.0,
            stability_f1=0.78,
        )
        
        latex = all_metrics.to_latex_table()
        
        assert r"\begin{table}" in latex
        assert "0.85" in latex  # Precision
        assert r"\end{table}" in latex


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
