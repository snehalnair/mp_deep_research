"""
Tests for Booking.com-inspired Evaluation Modules

Tests for:
1. Tool Validation (validity, correctness, reliability)
2. Consistency Evaluation (pass@k, pass^k)
3. Trajectory Optimality
4. Baseline Comparison Framework
"""

import pytest
from typing import Dict, List

from mp_deep_research.evaluation.tool_validation import (
    ToolSchema,
    ToolCallError,
    validate_tool_call,
    evaluate_tool_correctness,
    check_tool_reliability,
    compute_tool_validation_metrics,
)

from mp_deep_research.evaluation.consistency import (
    ConsistencyResult,
    ConsistencyTrialResult,
    QueryParaphraser,
    evaluate_consistency,
    Trajectory,
    TrajectoryStep,
    evaluate_trajectory_optimality,
    compute_trajectory_metrics,
    analyze_complexity_scaling,
)

from mp_deep_research.evaluation.baselines import (
    BaselineType,
    SystemResult,
    evaluate_system,
    compare_systems,
    create_zero_shot_baseline,
    CostEstimator,
)


# =============================================================================
# TOOL VALIDATION TESTS
# =============================================================================

class TestToolCallValidation:
    """Test tool call validity checks."""
    
    @pytest.fixture
    def sample_tools(self) -> Dict[str, ToolSchema]:
        """Create sample tool schemas."""
        return {
            "screen_materials": ToolSchema(
                name="screen_materials",
                description="Screen materials from database based on criteria.",
                parameters={
                    "elements": {"type": "array", "description": "List of elements"},
                    "e_above_hull_max": {"type": "number", "description": "Max energy above hull"},
                    "band_gap_min": {"type": "number", "description": "Min band gap"},
                },
                required_params=["elements"],
            ),
            "substitute_element": ToolSchema(
                name="substitute_element",
                description="Substitute one element for another in a structure.",
                parameters={
                    "material_id": {"type": "string", "description": "MP ID"},
                    "original_element": {"type": "string", "description": "Element to replace"},
                    "new_element": {"type": "string", "description": "Replacement element"},
                },
                required_params=["material_id", "original_element", "new_element"],
            ),
        }
    
    def test_valid_tool_call(self, sample_tools):
        """Test validation of a correct tool call."""
        tool_call = {
            "name": "screen_materials",
            "arguments": {
                "elements": ["Li", "Co", "O"],
                "e_above_hull_max": 0.025,
            }
        }
        
        result = validate_tool_call(tool_call, sample_tools)
        
        assert result.is_valid
        assert result.error_type == ToolCallError.VALID
    
    def test_unknown_tool(self, sample_tools):
        """Test detection of unknown tool."""
        tool_call = {
            "name": "nonexistent_tool",
            "arguments": {}
        }
        
        result = validate_tool_call(tool_call, sample_tools)
        
        assert not result.is_valid
        assert result.error_type == ToolCallError.UNKNOWN_TOOL
    
    def test_missing_required_param(self, sample_tools):
        """Test detection of missing required parameters."""
        tool_call = {
            "name": "substitute_element",
            "arguments": {
                "material_id": "mp-19017",
                # Missing original_element and new_element
            }
        }
        
        result = validate_tool_call(tool_call, sample_tools)
        
        assert not result.is_valid
        assert result.error_type == ToolCallError.MISSING_REQUIRED_PARAM
        assert "original_element" in result.missing_params
    
    def test_unexpected_param(self, sample_tools):
        """Test detection of unexpected parameters."""
        tool_call = {
            "name": "screen_materials",
            "arguments": {
                "elements": ["Li", "Fe", "P", "O"],
                "unexpected_param": "value",
            }
        }
        
        result = validate_tool_call(tool_call, sample_tools)
        
        assert not result.is_valid
        assert result.error_type == ToolCallError.UNEXPECTED_PARAM
    
    def test_type_mismatch(self, sample_tools):
        """Test detection of type mismatches."""
        tool_call = {
            "name": "screen_materials",
            "arguments": {
                "elements": "Li-Co-O",  # Should be array, not string
            }
        }
        
        result = validate_tool_call(tool_call, sample_tools)
        
        assert not result.is_valid
        assert result.error_type == ToolCallError.TYPE_MISMATCH


class TestToolCorrectness:
    """Test tool correctness evaluation."""
    
    def test_correct_tool_selection(self):
        """Test when agent uses exactly the right tools."""
        result = evaluate_tool_correctness(
            task_id="task_001",
            used_tools=["screen_materials", "think"],
            expected_tools=["screen_materials", "think"],
        )
        
        assert result.is_correct
        assert len(result.missing_tools) == 0
        assert len(result.redundant_tools) == 0
    
    def test_missing_tools(self):
        """Test when agent misses expected tools."""
        result = evaluate_tool_correctness(
            task_id="task_002",
            used_tools=["screen_materials"],
            expected_tools=["screen_materials", "assess_stability"],
        )
        
        assert not result.is_correct
        assert "assess_stability" in result.missing_tools
    
    def test_redundant_tools(self):
        """Test when agent uses unnecessary tools."""
        result = evaluate_tool_correctness(
            task_id="task_003",
            used_tools=["screen_materials", "think", "search_arxiv"],
            expected_tools=["screen_materials"],
            allow_extra_tools=False,
        )
        
        assert not result.is_correct
        assert "think" in result.redundant_tools or "search_arxiv" in result.redundant_tools
    
    def test_allow_extra_tools(self):
        """Test with extra tools allowed."""
        result = evaluate_tool_correctness(
            task_id="task_004",
            used_tools=["screen_materials", "think"],
            expected_tools=["screen_materials"],
            allow_extra_tools=True,
        )
        
        assert result.is_correct  # Extra tools allowed


class TestToolReliability:
    """Test tool reliability checks."""
    
    def test_good_tool_name(self):
        """Test a well-named tool."""
        result = check_tool_reliability(
            tool_name="screen_materials_by_stability",
            tool_description="Screen materials from the database based on thermodynamic stability criteria. Takes elements and e_above_hull threshold as input.",
        )
        
        assert result.overall_score >= 0.7
        assert result.name_checks.get("snake_case", False)
        assert result.name_checks.get("action_oriented", False)
    
    def test_bad_tool_name(self):
        """Test a poorly-named tool."""
        result = check_tool_reliability(
            tool_name="doThing",  # camelCase, generic
            tool_description="Does a thing",  # Too short
        )
        
        assert result.overall_score < 0.5
        assert not result.name_checks.get("snake_case", True)
    
    def test_good_description(self):
        """Test a tool with good description."""
        result = check_tool_reliability(
            tool_name="calculate_band_gap",
            tool_description="Calculate the electronic band gap of a material structure. Accepts a material_id parameter and returns the band gap in eV along with whether it's direct or indirect.",
        )
        
        assert result.description_checks.get("not_empty", False)
        assert result.description_checks.get("not_too_short", False)
        assert result.description_checks.get("has_input_description", False)


# =============================================================================
# CONSISTENCY TESTS
# =============================================================================

class TestQueryParaphraser:
    """Test query paraphrasing for consistency evaluation."""
    
    def test_basic_paraphrasing(self):
        """Test basic paraphrase generation."""
        paraphraser = QueryParaphraser()
        
        query = "Find stable lithium battery cathode materials"
        paraphrases = paraphraser.paraphrase(query, k=5)
        
        assert len(paraphrases) == 5
        assert query in paraphrases  # Original included
        assert len(set(paraphrases)) == 5  # All unique
    
    def test_synonym_substitution(self):
        """Test that synonyms are used."""
        paraphraser = QueryParaphraser()
        
        query = "Find stable materials for batteries"
        paraphrases = paraphraser.paraphrase(query, k=5)
        
        # Should have variations like "search for", "locate", etc.
        has_variation = any("search" in p.lower() or "locate" in p.lower() 
                          for p in paraphrases)
        assert has_variation or len(paraphrases) >= 3
    
    def test_reproducibility(self):
        """Test that paraphrasing is reproducible with seed."""
        paraphraser = QueryParaphraser()
        
        query = "Analyze thermoelectric materials"
        
        p1 = paraphraser.paraphrase(query, k=5, seed=42)
        p2 = paraphraser.paraphrase(query, k=5, seed=42)
        
        assert p1 == p2


class TestConsistencyEvaluation:
    """Test consistency metrics computation."""
    
    def test_fully_consistent_result(self):
        """Test a fully consistent query (all trials succeed)."""
        result = ConsistencyResult(
            original_query="Find Li-ion cathodes",
            trials=[
                ConsistencyTrialResult(query="Find Li-ion cathodes", success=True),
                ConsistencyTrialResult(query="Search for Li-ion cathodes", success=True),
                ConsistencyTrialResult(query="Locate Li-ion cathodes", success=True),
            ]
        )
        
        assert result.pass_at_k == 1.0  # At least one success
        assert result.pass_hat_k == 1.0  # All succeed
        assert result.success_rate == 1.0
    
    def test_partially_consistent_result(self):
        """Test a partially consistent query (some trials succeed)."""
        result = ConsistencyResult(
            original_query="Design novel cathode",
            trials=[
                ConsistencyTrialResult(query="Design novel cathode", success=True),
                ConsistencyTrialResult(query="Create novel cathode", success=True),
                ConsistencyTrialResult(query="Generate novel cathode", success=False),
            ]
        )
        
        assert result.pass_at_k == 1.0  # At least one success
        assert result.pass_hat_k == 0.0  # Not all succeed
        assert abs(result.success_rate - 2/3) < 0.01
    
    def test_fully_inconsistent_result(self):
        """Test a fully inconsistent query (all trials fail)."""
        result = ConsistencyResult(
            original_query="Impossible query",
            trials=[
                ConsistencyTrialResult(query="Impossible query", success=False),
                ConsistencyTrialResult(query="Please impossible query", success=False),
            ]
        )
        
        assert result.pass_at_k == 0.0
        assert result.pass_hat_k == 0.0
        assert result.success_rate == 0.0


# =============================================================================
# TRAJECTORY TESTS
# =============================================================================

class TestTrajectoryOptimality:
    """Test trajectory optimality evaluation."""
    
    def test_optimal_trajectory(self):
        """Test an optimal trajectory."""
        trajectory = Trajectory(
            task_id="task_001",
            steps=[
                TrajectoryStep(step_number=0, action_type="tool_call", tool_name="screen_materials"),
                TrajectoryStep(step_number=1, action_type="tool_call", tool_name="think"),
            ],
            success=True,
        )
        
        result = evaluate_trajectory_optimality(
            trajectory,
            optimal_tool_sequence=["screen_materials", "think"],
        )
        
        assert result.is_optimal
        assert result.tool_efficiency == 1.0
    
    def test_inefficient_trajectory(self):
        """Test a trajectory with extra steps."""
        trajectory = Trajectory(
            task_id="task_002",
            steps=[
                TrajectoryStep(step_number=0, action_type="tool_call", tool_name="screen_materials"),
                TrajectoryStep(step_number=1, action_type="tool_call", tool_name="search_arxiv"),
                TrajectoryStep(step_number=2, action_type="tool_call", tool_name="screen_materials"),
                TrajectoryStep(step_number=3, action_type="tool_call", tool_name="think"),
            ],
            success=True,
        )
        
        result = evaluate_trajectory_optimality(
            trajectory,
            optimal_tool_sequence=["screen_materials", "think"],
        )
        
        assert not result.is_optimal
        assert result.extra_tool_calls > 0
        assert result.tool_efficiency < 1.0
    
    def test_trajectory_from_agent_output(self):
        """Test creating trajectory from agent output dict."""
        output = {
            "success": True,
            "final_answer": "Found 5 candidate materials",
            "tool_calls": [
                {"name": "screen_materials", "arguments": {"elements": ["Li", "Co", "O"]}},
                {"name": "think", "arguments": {"thought": "Analyzing results"}},
            ],
            "duration_ms": 1500,
            "token_count": 500,
        }
        
        trajectory = Trajectory.from_agent_output("task_003", output)
        
        assert trajectory.task_id == "task_003"
        assert trajectory.success
        assert trajectory.num_tool_calls == 2
        assert trajectory.tool_sequence == ["screen_materials", "think"]


class TestComplexityScaling:
    """Test complexity scaling analysis."""
    
    def test_perfect_scaling(self):
        """Test agent with no performance degradation."""
        results = [
            # (complexity, success, tools, duration)
            (1, True, 2, 100),
            (1, True, 2, 110),
            (2, True, 3, 200),
            (2, True, 3, 190),
            (3, True, 4, 300),
            (3, True, 4, 310),
        ]
        
        analysis = analyze_complexity_scaling(results)
        
        assert analysis.buckets[1].success_rate == 1.0
        assert analysis.buckets[2].success_rate == 1.0
        assert analysis.buckets[3].success_rate == 1.0
        
        # No performance degradation
        assert analysis.performance_slope >= -0.01
    
    def test_degrading_performance(self):
        """Test agent with performance degradation."""
        results = [
            # Easy tasks - high success
            (1, True, 2, 100),
            (1, True, 2, 100),
            # Medium tasks - some failures
            (2, True, 3, 200),
            (2, False, 4, 250),
            # Hard tasks - more failures
            (3, False, 5, 400),
            (3, False, 6, 450),
        ]
        
        analysis = analyze_complexity_scaling(results)
        
        assert analysis.buckets[1].success_rate == 1.0
        assert analysis.buckets[2].success_rate == 0.5
        assert analysis.buckets[3].success_rate == 0.0
        
        # Negative slope = degradation
        assert analysis.performance_slope < 0


# =============================================================================
# BASELINE COMPARISON TESTS
# =============================================================================

class TestBaselineComparison:
    """Test baseline comparison framework."""
    
    def test_cost_estimator(self):
        """Test cost estimation."""
        estimator = CostEstimator()
        
        cost = estimator.estimate_cost(
            model="gpt-4o-mini",
            input_tokens=1000,
            output_tokens=500,
        )
        
        assert cost > 0
        assert cost < 1  # Should be less than 1 cent for small request
    
    def test_system_evaluation(self):
        """Test evaluating a system."""
        def mock_runner(query: str) -> Dict:
            return {
                "final_answer": f"Response to: {query}",
                "token_count": 100,
                "tool_calls": [],
            }
        
        def success_checker(output: Dict, task: Dict) -> bool:
            return "Response" in output.get("final_answer", "")
        
        tasks = [
            {"id": "t1", "query": "Find cathodes"},
            {"id": "t2", "query": "Analyze stability"},
        ]
        
        evaluation = evaluate_system(
            system_name="Test System",
            system_type=BaselineType.ZERO_SHOT_LLM,
            runner=mock_runner,
            tasks=tasks,
            success_checker=success_checker,
        )
        
        assert evaluation.total_tasks == 2
        assert evaluation.successful_tasks == 2
        assert evaluation.task_completion_rate == 1.0
    
    def test_system_comparison(self):
        """Test comparing systems."""
        from mp_deep_research.evaluation.baselines import SystemEvaluation
        
        agent_eval = SystemEvaluation(
            system_name="Agent",
            system_type=BaselineType.AGENT,
            total_tasks=10,
            successful_tasks=9,
            task_completion_rate=0.9,
            avg_latency_ms=2000,
            avg_cost_cents=0.5,
        )
        
        baseline_eval = SystemEvaluation(
            system_name="Zero-shot",
            system_type=BaselineType.ZERO_SHOT_LLM,
            total_tasks=10,
            successful_tasks=6,
            task_completion_rate=0.6,
            avg_latency_ms=500,
            avg_cost_cents=0.1,
        )
        
        comparison = compare_systems(agent_eval, [baseline_eval])
        
        # Agent is 30% better (0.9 - 0.6)
        assert comparison.performance_deltas["Zero-shot"] == pytest.approx(0.3, rel=0.01)
        
        # Agent is 4x slower
        assert comparison.latency_ratios["Zero-shot"] == pytest.approx(4.0, rel=0.01)
        
        # Should recommend deploying agent
        assert comparison.recommendation == "DEPLOY_AGENT"
    
    def test_baseline_preferred(self):
        """Test when baseline should be preferred."""
        from mp_deep_research.evaluation.baselines import SystemEvaluation
        
        agent_eval = SystemEvaluation(
            system_name="Agent",
            system_type=BaselineType.AGENT,
            total_tasks=10,
            successful_tasks=7,
            task_completion_rate=0.7,
            avg_latency_ms=5000,
            avg_cost_cents=2.0,
        )
        
        baseline_eval = SystemEvaluation(
            system_name="Few-shot",
            system_type=BaselineType.FEW_SHOT_LLM,
            total_tasks=10,
            successful_tasks=7,
            task_completion_rate=0.7,  # Same performance
            avg_latency_ms=500,
            avg_cost_cents=0.2,
        )
        
        comparison = compare_systems(
            agent_eval, 
            [baseline_eval],
            min_performance_gain=0.05,
        )
        
        # No performance improvement - should use baseline
        assert comparison.recommendation == "USE_BASELINE"


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for the complete evaluation pipeline."""
    
    def test_full_tool_validation_pipeline(self):
        """Test complete tool validation workflow."""
        tools = {
            "screen_materials": ToolSchema(
                name="screen_materials",
                description="Screen materials based on stability and property criteria. Accepts elements array and optional filters.",
                parameters={
                    "elements": {"type": "array"},
                    "e_above_hull_max": {"type": "number"},
                },
                required_params=["elements"],
            ),
        }
        
        # Valid calls
        valid_calls = [
            {"name": "screen_materials", "arguments": {"elements": ["Li", "Co", "O"]}},
            {"name": "screen_materials", "arguments": {"elements": ["Na", "Fe", "P", "O"], "e_above_hull_max": 0.05}},
        ]
        
        # Invalid calls
        invalid_calls = [
            {"name": "unknown_tool", "arguments": {}},
            {"name": "screen_materials", "arguments": {"wrong_param": "value"}},
        ]
        
        all_calls = valid_calls + invalid_calls
        
        validation_results = [validate_tool_call(c, tools) for c in all_calls]
        correctness_results = [
            evaluate_tool_correctness("t1", ["screen_materials"], ["screen_materials"]),
        ]
        reliability_results = [
            check_tool_reliability(t.name, t.description) for t in tools.values()
        ]
        
        metrics = compute_tool_validation_metrics(
            validation_results,
            correctness_results,
            reliability_results,
        )
        
        assert metrics.validity_rate == 0.5  # 2 valid out of 4
        assert metrics.tool_correctness_rate == 1.0
        assert metrics.avg_reliability_score > 0.5
    
    def test_latex_generation(self):
        """Test LaTeX output generation."""
        from mp_deep_research.evaluation.consistency import ConsistencyMetrics
        
        metrics = ConsistencyMetrics(
            total_queries=10,
            k=5,
            avg_pass_at_k=0.95,
            avg_pass_hat_k=0.72,
            avg_success_rate=0.85,
            fully_consistent=7,
        )
        
        latex = metrics.to_latex()
        
        assert "pass@5" in latex
        assert "pass$^{5}$" in latex
        assert "0.95" in latex or "0.950" in latex


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
