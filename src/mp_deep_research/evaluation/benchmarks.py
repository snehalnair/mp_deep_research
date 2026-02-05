"""
Benchmark Suites for Materials Project Deep Research

This module defines benchmark tasks for systematic evaluation:

1. Discovery Benchmark: Tasks that test materials retrieval from MP database
2. Innovation Benchmark: Tasks that test novel materials design workflow
3. End-to-End Benchmark: Complete research scenarios

Each benchmark includes:
- Task definition with clear success criteria
- Ground truth from DFT calculations or experiments
- Difficulty levels (easy, medium, hard)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
import json
import time
from pathlib import Path


class TaskDifficulty(Enum):
    """Task difficulty levels."""
    EASY = "easy"        # Single tool call, clear answer
    MEDIUM = "medium"    # Multiple tool calls, some reasoning
    HARD = "hard"        # Complex workflow, requires planning


class TaskType(Enum):
    """Types of benchmark tasks."""
    DISCOVERY = "discovery"
    INNOVATION = "innovation"
    COMPARISON = "comparison"
    ANALYSIS = "analysis"
    END_TO_END = "end_to_end"


@dataclass
class BenchmarkTask:
    """A single benchmark task."""
    task_id: str
    task_type: TaskType
    difficulty: TaskDifficulty
    
    # Task definition
    description: str
    user_query: str  # The actual query to send to the agent
    
    # Ground truth
    expected_materials: List[str] = field(default_factory=list)  # MP IDs
    expected_properties: Dict[str, Any] = field(default_factory=dict)
    expected_tool_sequence: List[str] = field(default_factory=list)
    success_criteria: str = ""
    
    # Metadata
    chemical_system: Optional[str] = None
    target_application: Optional[str] = None
    reference_doi: Optional[str] = None  # Paper reference if applicable
    
    # Scoring weights
    property_weight: float = 0.4
    material_weight: float = 0.4
    reasoning_weight: float = 0.2


@dataclass
class BenchmarkResult:
    """Result of running a benchmark task."""
    task_id: str
    success: bool
    
    # Scores (0-1)
    overall_score: float
    property_score: float
    material_score: float
    reasoning_score: float
    
    # Details
    materials_found: List[str]
    properties_extracted: Dict[str, Any]
    tool_calls_made: List[str]
    execution_time_seconds: float
    token_count: int
    
    # Agent output
    final_answer: str
    reasoning_trace: List[str]
    
    # Error info
    errors: List[str] = field(default_factory=list)


# =============================================================================
# DISCOVERY BENCHMARK
# =============================================================================

class DiscoveryBenchmark:
    """
    Benchmark suite for materials discovery tasks.
    
    Tests the agent's ability to:
    1. Query the MP database effectively
    2. Apply appropriate filters
    3. Rank and select candidates
    4. Identify data quality issues
    """
    
    def __init__(self):
        self.tasks = self._create_tasks()

    def get_tasks(self, difficulty: str | TaskDifficulty | None = None, limit: int | None = None) -> List[BenchmarkTask]:
        """Get tasks filtered by difficulty with an optional limit."""
        tasks = self.tasks
        if difficulty:
            if isinstance(difficulty, str):
                difficulty = TaskDifficulty(difficulty.lower())
            tasks = [t for t in tasks if t.difficulty == difficulty]
        if limit is not None:
            tasks = tasks[:limit]
        return tasks
    
    def _create_tasks(self) -> List[BenchmarkTask]:
        """Create the discovery benchmark tasks."""
        return [
            # ===== EASY: Single chemical system queries =====
            BenchmarkTask(
                task_id="disc_easy_001",
                task_type=TaskType.DISCOVERY,
                difficulty=TaskDifficulty.EASY,
                description="Find stable lithium ion battery cathode materials",
                user_query="Find me stable lithium-based cathode materials in the Li-Co-O system with energy above hull less than 25 meV/atom",
                expected_materials=["mp-22526", "mp-24850"],  # LiCoO2 polymorphs
                expected_properties={
                    "stability_threshold": 0.025,
                    "chemical_system": "Li-Co-O"
                },
                expected_tool_sequence=["batch_screen_materials"],
                success_criteria="Returns LiCoO2 (mp-22526) in top 5 results",
                chemical_system="Li-Co-O",
                target_application="Li-ion battery cathode",
            ),
            
            BenchmarkTask(
                task_id="disc_easy_002",
                task_type=TaskType.DISCOVERY,
                difficulty=TaskDifficulty.EASY,
                description="Find thermoelectric materials in Bi-Te system",
                user_query="Search for semiconducting materials in the Bi-Te system",
                expected_materials=["mp-34202"],  # Bi2Te3
                expected_properties={
                    "require_semiconductor": True,
                    "chemical_system": "Bi-Te"
                },
                expected_tool_sequence=["batch_screen_materials"],
                success_criteria="Returns Bi2Te3 as a top candidate",
                chemical_system="Bi-Te",
                target_application="Thermoelectric",
            ),
            
            BenchmarkTask(
                task_id="disc_easy_003",
                task_type=TaskType.DISCOVERY,
                difficulty=TaskDifficulty.EASY,
                description="Find transparent conducting oxides",
                user_query="Find stable oxide materials with band gap between 2.5 and 4.0 eV in the In-Sn-O system",
                expected_materials=["mp-20232"],  # In2O3
                expected_properties={
                    "band_gap_min": 2.5,
                    "band_gap_max": 4.0,
                    "chemical_system": "In-Sn-O"
                },
                expected_tool_sequence=["batch_screen_materials"],
                success_criteria="Returns In2O3 or ITO variants",
                chemical_system="In-Sn-O",
                target_application="Transparent conductor",
            ),
            
            # ===== MEDIUM: Multi-step queries with analysis =====
            BenchmarkTask(
                task_id="disc_med_001",
                task_type=TaskType.DISCOVERY,
                difficulty=TaskDifficulty.MEDIUM,
                description="Compare olivine cathode materials",
                user_query="Compare the stability and electronic properties of LiFePO4 and LiMnPO4. Which one is more promising for battery applications?",
                expected_materials=["mp-19017", "mp-18767"],  # LiFePO4, LiMnPO4
                expected_properties={
                    "compare_stability": True,
                    "compare_band_gap": True,
                },
                expected_tool_sequence=["batch_screen_materials", "analyze_candidate", "analyze_candidate"],
                success_criteria="Correctly identifies LiFePO4 as more stable and discusses trade-offs",
                chemical_system="Li-Fe-Mn-P-O",
                target_application="Li-ion battery cathode",
            ),
            
            BenchmarkTask(
                task_id="disc_med_002",
                task_type=TaskType.DISCOVERY,
                difficulty=TaskDifficulty.MEDIUM,
                description="Perovskite solar cell material search with constraints",
                user_query="Find lead-free perovskite materials (systems containing Cs or MA with halides) suitable for solar cells. Band gap should be 1.1-1.7 eV.",
                expected_materials=["mp-23037"],  # CsSnI3
                expected_properties={
                    "band_gap_range": [1.1, 1.7],
                    "exclude_elements": ["Pb"],
                },
                expected_tool_sequence=["batch_screen_materials"],
                success_criteria="Returns lead-free perovskites with appropriate band gaps",
                target_application="Solar cell absorber",
            ),
            
            BenchmarkTask(
                task_id="disc_med_003",
                task_type=TaskType.DISCOVERY,
                difficulty=TaskDifficulty.MEDIUM,
                description="Generate and analyze phase diagram",
                user_query="Generate a phase diagram for the Li-Mn-O system and identify the stable phases",
                expected_materials=["mp-19017", "mp-18767", "mp-25584"],
                expected_properties={
                    "stable_phases_count": ">=3",
                },
                expected_tool_sequence=["generate_phase_diagram"],
                success_criteria="Correctly generates phase diagram and identifies key phases",
                chemical_system="Li-Mn-O",
                target_application="Battery materials",
            ),
            
            # ===== HARD: Complex multi-criteria optimization =====
            BenchmarkTask(
                task_id="disc_hard_001",
                task_type=TaskType.DISCOVERY,
                difficulty=TaskDifficulty.HARD,
                description="Find cheaper cobalt-free cathode alternatives",
                user_query="Find stable lithium cathode materials that do NOT contain cobalt, have voltage > 3.5V, and are cost-effective (no rare earth elements). Consider Li-Fe-Mn-P-O and Li-Ni-Mn-O systems.",
                expected_materials=["mp-19017", "mp-756198"],  # LiFePO4, LNMO variants
                expected_properties={
                    "exclude_elements": ["Co"],
                    "voltage_min": 3.5,
                },
                expected_tool_sequence=["batch_screen_materials", "batch_screen_materials", "analyze_candidate"],
                success_criteria="Identifies LiFePO4 and discusses alternatives with reasoning",
                target_application="Cost-effective Li-ion cathode",
            ),
            
            BenchmarkTask(
                task_id="disc_hard_002",
                task_type=TaskType.DISCOVERY,
                difficulty=TaskDifficulty.HARD,
                description="Solid electrolyte discovery",
                user_query="Find superionic conductors in the Li-La-Zr-O system that could work as solid electrolytes. Prioritize materials with ICSD validation.",
                expected_materials=["mp-35016"],  # LLZO
                expected_properties={
                    "prioritize_experimental": True,
                    "target_property": "ionic_conductivity",
                },
                expected_tool_sequence=["batch_screen_materials", "analyze_candidate", "identify_data_gaps"],
                success_criteria="Returns LLZO and notes data quality/experimental status",
                chemical_system="Li-La-Zr-O",
                target_application="Solid electrolyte",
            ),
            
            BenchmarkTask(
                task_id="disc_hard_003",
                task_type=TaskType.DISCOVERY,
                difficulty=TaskDifficulty.HARD,
                description="Multi-criteria photovoltaic optimization",
                user_query="Design a materials search strategy for finding earth-abundant, stable photovoltaic absorbers. Search Cu-Zn-Sn-S and Cu-Bi-S systems. Explain your methodology and rank candidates.",
                expected_materials=["mp-27502"],  # CZTS
                expected_properties={
                    "earth_abundant": True,
                    "band_gap_range": [1.0, 1.5],
                    "stability": "<50 meV",
                },
                expected_tool_sequence=["batch_screen_materials", "batch_screen_materials", "analyze_candidate"],
                success_criteria="Systematic search with clear ranking criteria and reasoning",
                target_application="Solar cell absorber",
            ),
        ]
    
    def get_tasks_by_difficulty(self, difficulty: TaskDifficulty) -> List[BenchmarkTask]:
        """Get tasks filtered by difficulty."""
        return [t for t in self.tasks if t.difficulty == difficulty]
    
    def __len__(self) -> int:
        return len(self.tasks)


# =============================================================================
# INNOVATION BENCHMARK
# =============================================================================

class InnovationBenchmark:
    """
    Benchmark suite for materials innovation tasks.
    
    Tests the agent's ability to:
    1. Propose sensible chemical substitutions
    2. Run ML relaxation workflows
    3. Assess stability of novel materials
    4. Make appropriate recommendations
    """
    
    def __init__(self):
        self.tasks = self._create_tasks()
        # Ground truth energies from DFT (for validation)
        self.ground_truth_energies = self._load_ground_truth()

    def get_tasks(self, difficulty: str | TaskDifficulty | None = None, limit: int | None = None) -> List[BenchmarkTask]:
        """Get tasks filtered by difficulty with an optional limit."""
        tasks = self.tasks
        if difficulty:
            if isinstance(difficulty, str):
                difficulty = TaskDifficulty(difficulty.lower())
            tasks = [t for t in tasks if t.difficulty == difficulty]
        if limit is not None:
            tasks = tasks[:limit]
        return tasks
    
    def _create_tasks(self) -> List[BenchmarkTask]:
        """Create the innovation benchmark tasks."""
        return [
            # ===== EASY: Simple single substitutions =====
            BenchmarkTask(
                task_id="innov_easy_001",
                task_type=TaskType.INNOVATION,
                difficulty=TaskDifficulty.EASY,
                description="Sodium analogue of LiFePO4",
                user_query="Create a sodium version of LiFePO4 (mp-19017) by substituting Li with Na. Relax the structure and assess stability.",
                expected_materials=["mp-19017"],  # Parent material
                expected_properties={
                    "substitution": {"Li": "Na"},
                    "expected_e_hull_range": [0, 100],  # meV
                },
                expected_tool_sequence=["substitute_species", "relax_structure_m3gnet", "assess_stability"],
                success_criteria="Creates NaFePO4, relaxes, and correctly predicts metastability",
                chemical_system="Na-Fe-P-O",
                target_application="Na-ion battery cathode",
            ),
            
            BenchmarkTask(
                task_id="innov_easy_002",
                task_type=TaskType.INNOVATION,
                difficulty=TaskDifficulty.EASY,
                description="Manganese variant of LiCoO2",
                user_query="What happens if we replace Co with Mn in LiCoO2 (mp-22526)? Run the innovation workflow.",
                expected_materials=["mp-22526"],
                expected_properties={
                    "substitution": {"Co": "Mn"},
                },
                expected_tool_sequence=["substitute_species", "relax_structure_m3gnet", "assess_stability"],
                success_criteria="Creates LiMnO2, assesses stability correctly",
                chemical_system="Li-Mn-O",
                target_application="Li-ion battery cathode",
            ),
            
            # ===== MEDIUM: Multiple substitutions or complex assessment =====
            BenchmarkTask(
                task_id="innov_med_001",
                task_type=TaskType.INNOVATION,
                difficulty=TaskDifficulty.MEDIUM,
                description="Double substitution in olivine",
                user_query="Try replacing both Li with Na AND Fe with Mn in LiFePO4. Is this double substitution stable?",
                expected_materials=["mp-19017"],
                expected_properties={
                    "substitution": {"Li": "Na", "Fe": "Mn"},
                },
                expected_tool_sequence=["substitute_species", "relax_structure_m3gnet", "assess_stability"],
                success_criteria="Creates NaMnPO4, correctly assesses as less stable than single substitutions",
                chemical_system="Na-Mn-P-O",
            ),
            
            BenchmarkTask(
                task_id="innov_med_002",
                task_type=TaskType.INNOVATION,
                difficulty=TaskDifficulty.MEDIUM,
                description="Systematic transition metal substitution",
                user_query="Starting from LiFePO4, systematically test replacing Fe with Mn, Co, and Ni. Compare the stability of all three variants.",
                expected_materials=["mp-19017"],
                expected_properties={
                    "substitutions_to_test": [
                        {"Fe": "Mn"},
                        {"Fe": "Co"},
                        {"Fe": "Ni"},
                    ],
                },
                expected_tool_sequence=["run_innovation_flow"] * 3 + ["think"],
                success_criteria="Tests all three, compares stabilities, identifies most stable variant",
                chemical_system="Li-TM-P-O",
            ),
            
            # ===== HARD: Full research workflow =====
            BenchmarkTask(
                task_id="innov_hard_001",
                task_type=TaskType.INNOVATION,
                difficulty=TaskDifficulty.HARD,
                description="Design cheaper cathode by element replacement",
                user_query="""I want to find a cheaper alternative to LiCoO2 for batteries. 
                Cobalt is expensive and unethical to mine. 
                Starting from LiCoO2 (mp-22526), explore replacing Co with Fe, Mn, and Ni.
                For the most promising candidate, also try mixing (e.g., Co0.5Mn0.5).
                Provide a final recommendation with scientific justification.""",
                expected_materials=["mp-22526"],
                expected_properties={
                    "must_compare": True,
                    "must_justify": True,
                },
                expected_tool_sequence=[
                    "run_innovation_flow", "run_innovation_flow", 
                    "run_innovation_flow", "think"
                ],
                success_criteria="Systematic comparison, correct stability assessment, justified recommendation",
                target_application="Cost-effective cathode",
            ),
            
            BenchmarkTask(
                task_id="innov_hard_002",
                task_type=TaskType.INNOVATION,
                difficulty=TaskDifficulty.HARD,
                description="Ionic radius-guided optimization",
                user_query="""LiFePO4 has excellent stability but low voltage (~3.4V). 
                Can we increase the voltage by substituting Fe with elements that have higher redox potentials?
                Consider the ionic radius compatibility (Fe2+ = 0.78Å).
                Test Mn, Co, Ni, and V substitutions.
                Analyze which balances voltage improvement with structural stability.""",
                expected_materials=["mp-19017"],
                expected_properties={
                    "ionic_radius_analysis": True,
                    "voltage_consideration": True,
                },
                expected_tool_sequence=["run_innovation_flow"] * 4 + ["think"],
                success_criteria="Considers ionic radii, predicts stability, discusses voltage-stability tradeoff",
                target_application="High-voltage cathode",
            ),
            
            BenchmarkTask(
                task_id="innov_hard_003",
                task_type=TaskType.INNOVATION,
                difficulty=TaskDifficulty.HARD,
                description="Novel solid electrolyte design",
                user_query="""Design a novel lithium superionic conductor based on LLZO (Li7La3Zr2O12).
                Explore Al and Ta doping at the Zr site to improve ionic conductivity.
                Assess the structural stability of each doped variant.
                Which doping strategy looks most promising?""",
                expected_materials=["mp-35016"],  # LLZO
                expected_properties={
                    "doping_elements": ["Al", "Ta"],
                    "target_property": "ionic_conductivity",
                },
                expected_tool_sequence=["substitute_species", "relax_structure_m3gnet", "assess_stability"] * 2,
                success_criteria="Systematic doping study with stability assessment and recommendation",
                chemical_system="Li-La-Zr-Al-Ta-O",
                target_application="Solid electrolyte",
            ),
        ]
    
    def _load_ground_truth(self) -> Dict[str, Dict]:
        """
        Load DFT ground truth for validation.
        
        In a real implementation, this would load from a curated dataset.
        Here we provide representative values from literature.
        """
        return {
            # NaFePO4 (from Li→Na substitution in LiFePO4)
            "NaFePO4": {
                "dft_e_above_hull_mev": 45,  # Metastable
                "dft_energy_per_atom_ev": -6.832,
                "volume_change_percent": 12.5,  # Na is larger than Li
            },
            # LiMnPO4 (from Fe→Mn substitution in LiFePO4)
            "LiMnPO4": {
                "dft_e_above_hull_mev": 0,  # Stable
                "dft_energy_per_atom_ev": -6.754,
                "volume_change_percent": 2.3,
            },
            # LiMnO2 (from Co→Mn in LiCoO2)
            "LiMnO2": {
                "dft_e_above_hull_mev": 15,  # Near stable
                "dft_energy_per_atom_ev": -5.621,
                "volume_change_percent": 3.8,
            },
            # LiCoO2 (reference)
            "LiCoO2": {
                "dft_e_above_hull_mev": 0,
                "dft_energy_per_atom_ev": -5.847,
                "volume_change_percent": 0,
            },
        }
    
    def get_ground_truth_for_formula(self, formula: str) -> Optional[Dict]:
        """Get DFT ground truth for a formula if available."""
        return self.ground_truth_energies.get(formula)
    
    def __len__(self) -> int:
        return len(self.tasks)


# =============================================================================
# BENCHMARK RUNNER
# =============================================================================

@dataclass
class BenchmarkSuiteResult:
    """Results from running a complete benchmark suite."""
    suite_name: str
    total_tasks: int
    completed_tasks: int
    
    # Aggregate scores
    avg_overall_score: float
    avg_property_score: float
    avg_material_score: float
    avg_reasoning_score: float
    
    # By difficulty
    scores_by_difficulty: Dict[str, float]
    
    # Individual results
    task_results: List[BenchmarkResult]
    
    # Timing
    total_time_seconds: float
    
    def to_dict(self) -> Dict:
        return {
            "suite_name": self.suite_name,
            "total_tasks": self.total_tasks,
            "completed_tasks": self.completed_tasks,
            "completion_rate": round(self.completed_tasks / self.total_tasks, 4),
            "avg_overall_score": round(self.avg_overall_score, 4),
            "avg_property_score": round(self.avg_property_score, 4),
            "avg_material_score": round(self.avg_material_score, 4),
            "avg_reasoning_score": round(self.avg_reasoning_score, 4),
            "scores_by_difficulty": {
                k: round(v, 4) for k, v in self.scores_by_difficulty.items()
            },
            "total_time_seconds": round(self.total_time_seconds, 2),
        }


def run_benchmark_suite(
    benchmark: DiscoveryBenchmark | InnovationBenchmark,
    agent_runner: Callable[[str], Dict],
    evaluator: Optional[Callable[[BenchmarkTask, Dict], BenchmarkResult]] = None,
    max_tasks: Optional[int] = None,
) -> BenchmarkSuiteResult:
    """
    Run a complete benchmark suite against an agent.
    
    Args:
        benchmark: The benchmark suite to run
        agent_runner: Function that takes a query and returns agent output
        evaluator: Optional custom evaluator function
        max_tasks: Optional limit on number of tasks to run
    
    Returns:
        BenchmarkSuiteResult with all scores and details
    """
    tasks = benchmark.tasks[:max_tasks] if max_tasks else benchmark.tasks
    results = []
    total_time = 0.0
    
    for task in tasks:
        start_time = time.time()
        
        try:
            # Run agent
            agent_output = agent_runner(task.user_query)
            execution_time = time.time() - start_time
            
            # Evaluate
            if evaluator:
                result = evaluator(task, agent_output)
            else:
                result = _default_evaluate(task, agent_output, execution_time)
            
            results.append(result)
            
        except Exception as e:
            # Record failure
            results.append(BenchmarkResult(
                task_id=task.task_id,
                success=False,
                overall_score=0.0,
                property_score=0.0,
                material_score=0.0,
                reasoning_score=0.0,
                materials_found=[],
                properties_extracted={},
                tool_calls_made=[],
                execution_time_seconds=time.time() - start_time,
                token_count=0,
                final_answer="",
                reasoning_trace=[],
                errors=[str(e)],
            ))
        
        total_time += time.time() - start_time
    
    # Aggregate scores
    completed = [r for r in results if r.success]
    
    scores_by_difficulty = {}
    for diff in TaskDifficulty:
        diff_tasks = [t.task_id for t in tasks if t.difficulty == diff]
        diff_results = [r for r in completed if r.task_id in diff_tasks]
        if diff_results:
            scores_by_difficulty[diff.value] = np.mean([r.overall_score for r in diff_results])
        else:
            scores_by_difficulty[diff.value] = 0.0
    
    return BenchmarkSuiteResult(
        suite_name=type(benchmark).__name__,
        total_tasks=len(tasks),
        completed_tasks=len(completed),
        avg_overall_score=np.mean([r.overall_score for r in completed]) if completed else 0.0,
        avg_property_score=np.mean([r.property_score for r in completed]) if completed else 0.0,
        avg_material_score=np.mean([r.material_score for r in completed]) if completed else 0.0,
        avg_reasoning_score=np.mean([r.reasoning_score for r in completed]) if completed else 0.0,
        scores_by_difficulty=scores_by_difficulty,
        task_results=results,
        total_time_seconds=total_time,
    )


def _default_evaluate(
    task: BenchmarkTask,
    agent_output: Dict,
    execution_time: float
) -> BenchmarkResult:
    """Default evaluator for benchmark tasks."""
    
    # Extract relevant info from agent output
    final_answer = agent_output.get("final_answer", "")
    tool_calls = agent_output.get("tool_calls", [])
    reasoning = agent_output.get("reasoning", [])
    materials = agent_output.get("materials_mentioned", [])
    properties = agent_output.get("properties", {})
    
    # Score materials found
    expected_set = set(task.expected_materials)
    found_set = set(materials)
    if expected_set:
        material_score = len(expected_set & found_set) / len(expected_set)
    else:
        material_score = 1.0  # No specific materials expected
    
    # Score properties
    property_score = 0.0
    if task.expected_properties:
        matches = 0
        for key, expected in task.expected_properties.items():
            if key in properties:
                actual = properties[key]
                if isinstance(expected, list) and len(expected) == 2:
                    # Range check
                    if expected[0] <= actual <= expected[1]:
                        matches += 1
                elif actual == expected:
                    matches += 1
        property_score = matches / len(task.expected_properties)
    else:
        property_score = 1.0
    
    # Score tool sequence (simplified)
    tool_names = [call.get("name", "") for call in tool_calls]
    expected_tools = set(task.expected_tool_sequence)
    used_tools = set(tool_names)
    tool_score = len(expected_tools & used_tools) / max(1, len(expected_tools))
    
    # Reasoning score (placeholder - would use LLM-as-Judge in production)
    reasoning_score = 0.7 if reasoning else 0.3
    
    # Check success criteria
    success = task.success_criteria.lower() in final_answer.lower() if task.success_criteria else True
    
    # Weighted overall score
    overall_score = (
        task.material_weight * material_score +
        task.property_weight * property_score +
        task.reasoning_weight * reasoning_score
    )
    
    return BenchmarkResult(
        task_id=task.task_id,
        success=success and overall_score > 0.5,
        overall_score=overall_score,
        property_score=property_score,
        material_score=material_score,
        reasoning_score=reasoning_score,
        materials_found=materials,
        properties_extracted=properties,
        tool_calls_made=tool_names,
        execution_time_seconds=execution_time,
        token_count=agent_output.get("token_count", 0),
        final_answer=final_answer,
        reasoning_trace=reasoning,
    )


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def export_benchmark_to_json(
    benchmark: DiscoveryBenchmark | InnovationBenchmark,
    output_path: str | Path
) -> None:
    """Export benchmark tasks to JSON for reproducibility."""
    tasks_data = []
    for task in benchmark.tasks:
        tasks_data.append({
            "task_id": task.task_id,
            "task_type": task.task_type.value,
            "difficulty": task.difficulty.value,
            "description": task.description,
            "user_query": task.user_query,
            "expected_materials": task.expected_materials,
            "expected_properties": task.expected_properties,
            "expected_tool_sequence": task.expected_tool_sequence,
            "success_criteria": task.success_criteria,
            "chemical_system": task.chemical_system,
            "target_application": task.target_application,
        })
    
    with open(output_path, "w") as f:
        json.dump(tasks_data, f, indent=2)


def load_benchmark_from_json(json_path: str | Path) -> List[BenchmarkTask]:
    """Load benchmark tasks from JSON."""
    with open(json_path) as f:
        tasks_data = json.load(f)
    
    tasks = []
    for data in tasks_data:
        tasks.append(BenchmarkTask(
            task_id=data["task_id"],
            task_type=TaskType(data["task_type"]),
            difficulty=TaskDifficulty(data["difficulty"]),
            description=data["description"],
            user_query=data["user_query"],
            expected_materials=data.get("expected_materials", []),
            expected_properties=data.get("expected_properties", {}),
            expected_tool_sequence=data.get("expected_tool_sequence", []),
            success_criteria=data.get("success_criteria", ""),
            chemical_system=data.get("chemical_system"),
            target_application=data.get("target_application"),
        ))
    
    return tasks
