"""
Evaluation Metrics for Materials Project Deep Research

This module defines metrics for evaluating:
1. Screening tool performance (precision, recall, ranking quality)
2. Innovation pipeline accuracy (energy predictions, stability)
3. Agent reasoning and planning quality
4. End-to-end task completion

References:
- Materials science accuracy metrics: MAE, RMSE for energy predictions
- Ranking metrics: NDCG, MRR for candidate prioritization
- LLM evaluation: LLM-as-Judge patterns for reasoning quality
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import numpy as np
from collections import defaultdict
import json


# =============================================================================
# SECTION 1: DATA STRUCTURES
# =============================================================================

class StabilityVerdict(Enum):
    """Ground truth stability classifications."""
    STABLE = "stable"           # E_above_hull < 25 meV
    METASTABLE = "metastable"   # 25 <= E_above_hull < 50 meV  
    MARGINAL = "marginal"       # 50 <= E_above_hull < 100 meV
    UNSTABLE = "unstable"       # E_above_hull >= 100 meV


@dataclass
class MaterialPrediction:
    """Single material prediction from the system."""
    material_id: str
    formula: str
    predicted_stable: bool
    predicted_e_hull: Optional[float] = None  # meV/atom
    predicted_band_gap: Optional[float] = None  # eV
    rank_position: Optional[int] = None
    confidence_score: Optional[float] = None


@dataclass
class MaterialGroundTruth:
    """Ground truth for a material (from DFT or experiment)."""
    material_id: str
    formula: str
    is_stable: bool
    e_above_hull: float  # meV/atom (DFT value)
    band_gap: Optional[float] = None  # eV
    is_synthesized: bool = False  # Experimentally verified
    source: str = "MP-DFT"  # "MP-DFT", "ICSD", "Experiment"


@dataclass 
class AgentTrace:
    """Trace of agent execution for evaluation."""
    task_id: str
    task_description: str
    tool_calls: List[Dict[str, Any]]
    reasoning_steps: List[str]
    final_answer: str
    execution_time_seconds: float
    token_count: int
    success: bool


# =============================================================================
# SECTION 2: SCREENING METRICS
# =============================================================================

@dataclass
class ScreeningMetrics:
    """
    Metrics for evaluating the screening/discovery tools.
    
    Measures how well the system identifies promising materials from the database.
    """
    # Classification metrics
    precision: float = 0.0  # TP / (TP + FP)
    recall: float = 0.0     # TP / (TP + FN)  
    f1_score: float = 0.0   # 2 * (P * R) / (P + R)
    
    # Ranking metrics
    ndcg_at_5: float = 0.0   # Normalized Discounted Cumulative Gain @ k=5
    ndcg_at_10: float = 0.0  # NDCG @ k=10
    mrr: float = 0.0         # Mean Reciprocal Rank
    
    # Coverage metrics
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    true_negatives: int = 0
    
    # Domain-specific
    stable_materials_found: int = 0
    total_stable_in_ground_truth: int = 0
    avg_e_hull_of_recommendations: float = 0.0  # meV
    
    # Detailed breakdown
    confusion_matrix: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1_score": round(self.f1_score, 4),
            "ndcg@5": round(self.ndcg_at_5, 4),
            "ndcg@10": round(self.ndcg_at_10, 4),
            "mrr": round(self.mrr, 4),
            "stable_materials_found": self.stable_materials_found,
            "total_stable_in_ground_truth": self.total_stable_in_ground_truth,
            "avg_e_hull_of_recommendations_mev": round(self.avg_e_hull_of_recommendations, 2),
        }


def compute_screening_metrics(
    predictions: List[MaterialPrediction],
    ground_truth: List[MaterialGroundTruth],
    stability_threshold_mev: float = 25.0
) -> ScreeningMetrics:
    """
    Compute screening metrics by comparing predictions to ground truth.
    
    Args:
        predictions: List of predicted materials with rankings
        ground_truth: List of ground truth materials
        stability_threshold_mev: Threshold for "stable" classification (default 25 meV)
    
    Returns:
        ScreeningMetrics with all computed values
    """
    metrics = ScreeningMetrics()
    
    # Build lookup tables
    gt_lookup = {gt.material_id: gt for gt in ground_truth}
    pred_lookup = {p.material_id: p for p in predictions}
    
    # Identify stable materials in ground truth
    stable_gt_ids = {
        gt.material_id for gt in ground_truth 
        if gt.e_above_hull < stability_threshold_mev
    }
    metrics.total_stable_in_ground_truth = len(stable_gt_ids)
    
    # Classification metrics
    tp, fp, fn, tn = 0, 0, 0, 0
    e_hull_values = []
    
    for pred in predictions:
        if pred.material_id in gt_lookup:
            gt = gt_lookup[pred.material_id]
            is_actually_stable = gt.e_above_hull < stability_threshold_mev
            
            if pred.predicted_stable and is_actually_stable:
                tp += 1
                e_hull_values.append(gt.e_above_hull)
            elif pred.predicted_stable and not is_actually_stable:
                fp += 1
            elif not pred.predicted_stable and is_actually_stable:
                fn += 1
            else:
                tn += 1
    
    # Also count stable materials we missed entirely
    predicted_ids = {p.material_id for p in predictions}
    for gt_id in stable_gt_ids:
        if gt_id not in predicted_ids:
            fn += 1
    
    metrics.true_positives = tp
    metrics.false_positives = fp
    metrics.false_negatives = fn
    metrics.true_negatives = tn
    metrics.stable_materials_found = tp
    
    # Precision, Recall, F1
    if tp + fp > 0:
        metrics.precision = tp / (tp + fp)
    if tp + fn > 0:
        metrics.recall = tp / (tp + fn)
    if metrics.precision + metrics.recall > 0:
        metrics.f1_score = 2 * (metrics.precision * metrics.recall) / (metrics.precision + metrics.recall)
    
    # Average E_hull of recommendations
    if e_hull_values:
        metrics.avg_e_hull_of_recommendations = np.mean(e_hull_values)
    
    # Ranking metrics (NDCG, MRR)
    metrics.ndcg_at_5 = _compute_ndcg(predictions, gt_lookup, k=5, threshold=stability_threshold_mev)
    metrics.ndcg_at_10 = _compute_ndcg(predictions, gt_lookup, k=10, threshold=stability_threshold_mev)
    metrics.mrr = _compute_mrr(predictions, stable_gt_ids)
    
    # Confusion matrix
    metrics.confusion_matrix = {
        "TP": tp, "FP": fp, "FN": fn, "TN": tn
    }
    
    return metrics


def _compute_ndcg(
    predictions: List[MaterialPrediction],
    gt_lookup: Dict[str, MaterialGroundTruth],
    k: int,
    threshold: float
) -> float:
    """
    Compute Normalized Discounted Cumulative Gain at k.
    
    Relevance score: 2 if stable, 1 if metastable, 0 otherwise
    """
    # Sort predictions by rank
    sorted_preds = sorted(predictions, key=lambda x: x.rank_position or 999)[:k]
    
    # Compute DCG
    dcg = 0.0
    for i, pred in enumerate(sorted_preds):
        if pred.material_id in gt_lookup:
            gt = gt_lookup[pred.material_id]
            rel = 2 if gt.e_above_hull < 25 else (1 if gt.e_above_hull < 50 else 0)
            dcg += rel / np.log2(i + 2)
    
    # Compute ideal DCG
    all_relevances = []
    for gt in gt_lookup.values():
        rel = 2 if gt.e_above_hull < 25 else (1 if gt.e_above_hull < 50 else 0)
        all_relevances.append(rel)
    
    all_relevances.sort(reverse=True)
    idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(all_relevances[:k]))
    
    return dcg / idcg if idcg > 0 else 0.0


def _compute_mrr(
    predictions: List[MaterialPrediction],
    relevant_ids: set
) -> float:
    """Compute Mean Reciprocal Rank."""
    sorted_preds = sorted(predictions, key=lambda x: x.rank_position or 999)
    
    for i, pred in enumerate(sorted_preds):
        if pred.material_id in relevant_ids:
            return 1.0 / (i + 1)
    
    return 0.0


# =============================================================================
# SECTION 3: INNOVATION METRICS
# =============================================================================

@dataclass
class InnovationMetrics:
    """
    Metrics for evaluating the innovation/generative tools.
    
    Measures accuracy of:
    - Structure relaxation (M3GNet)
    - Energy predictions
    - Stability assessments
    - Novel material proposals
    """
    # Energy prediction accuracy
    energy_mae: float = 0.0      # Mean Absolute Error (eV/atom)
    energy_rmse: float = 0.0     # Root Mean Squared Error (eV/atom)
    energy_mae_mev: float = 0.0  # MAE in meV/atom (more intuitive)
    
    # Stability prediction accuracy
    stability_accuracy: float = 0.0
    stability_precision: float = 0.0
    stability_recall: float = 0.0
    stability_f1: float = 0.0
    
    # Volume prediction (structure relaxation quality)
    volume_mae_percent: float = 0.0  # MAE in % volume change
    
    # Innovation success rate
    total_substitutions_attempted: int = 0
    successful_relaxations: int = 0
    predicted_stable_count: int = 0
    confirmed_stable_count: int = 0  # If validated by DFT
    
    # Confidence calibration
    calibration_error: float = 0.0  # Expected Calibration Error
    
    # Breakdown by substitution type
    success_by_substitution_type: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "energy_mae_ev": round(self.energy_mae, 4),
            "energy_mae_mev": round(self.energy_mae_mev, 2),
            "energy_rmse_ev": round(self.energy_rmse, 4),
            "stability_accuracy": round(self.stability_accuracy, 4),
            "stability_f1": round(self.stability_f1, 4),
            "volume_mae_percent": round(self.volume_mae_percent, 2),
            "total_substitutions": self.total_substitutions_attempted,
            "successful_relaxations": self.successful_relaxations,
            "predicted_stable": self.predicted_stable_count,
            "confirmed_stable": self.confirmed_stable_count,
            "innovation_success_rate": round(
                self.confirmed_stable_count / max(1, self.predicted_stable_count), 4
            ),
        }


@dataclass
class InnovationPrediction:
    """Prediction from an innovation workflow."""
    parent_id: str
    substitutions: Dict[str, str]
    new_formula: str
    predicted_energy_per_atom: float  # eV/atom
    predicted_e_above_hull: float  # meV/atom
    predicted_stable: bool
    volume_change_percent: float
    relaxation_converged: bool
    confidence: Optional[float] = None


@dataclass
class InnovationGroundTruth:
    """Ground truth for an innovation task (from DFT)."""
    parent_id: str
    substitutions: Dict[str, str]
    new_formula: str
    dft_energy_per_atom: float  # eV/atom
    dft_e_above_hull: float  # meV/atom
    is_stable: bool  # DFT-determined stability
    dft_volume_change_percent: float
    exists_in_database: bool = False


def compute_innovation_metrics(
    predictions: List[InnovationPrediction],
    ground_truth: List[InnovationGroundTruth],
    stability_threshold_mev: float = 25.0
) -> InnovationMetrics:
    """
    Compute innovation metrics by comparing ML predictions to DFT ground truth.
    
    Args:
        predictions: List of innovation predictions (M3GNet results)
        ground_truth: List of DFT ground truth results
        stability_threshold_mev: Threshold for stability classification
    
    Returns:
        InnovationMetrics with all computed values
    """
    metrics = InnovationMetrics()
    
    # Build lookup by (parent_id, substitutions_key)
    def make_key(parent_id: str, subs: Dict[str, str]) -> str:
        return f"{parent_id}|{json.dumps(subs, sort_keys=True)}"
    
    gt_lookup = {make_key(gt.parent_id, gt.substitutions): gt for gt in ground_truth}
    
    # Collect paired data for metrics
    energy_errors = []
    volume_errors = []
    stability_pairs = []  # (predicted, actual)
    
    metrics.total_substitutions_attempted = len(predictions)
    
    for pred in predictions:
        key = make_key(pred.parent_id, pred.substitutions)
        
        if pred.relaxation_converged:
            metrics.successful_relaxations += 1
        
        if pred.predicted_stable:
            metrics.predicted_stable_count += 1
        
        if key in gt_lookup:
            gt = gt_lookup[key]
            
            # Energy error
            energy_error = pred.predicted_energy_per_atom - gt.dft_energy_per_atom
            energy_errors.append(energy_error)
            
            # Volume error
            vol_error = abs(pred.volume_change_percent - gt.dft_volume_change_percent)
            volume_errors.append(vol_error)
            
            # Stability classification
            predicted_stable = pred.predicted_e_above_hull < stability_threshold_mev
            actual_stable = gt.dft_e_above_hull < stability_threshold_mev
            stability_pairs.append((predicted_stable, actual_stable))
            
            if predicted_stable and actual_stable:
                metrics.confirmed_stable_count += 1
    
    # Compute energy metrics
    if energy_errors:
        energy_errors = np.array(energy_errors)
        metrics.energy_mae = float(np.mean(np.abs(energy_errors)))
        metrics.energy_rmse = float(np.sqrt(np.mean(energy_errors ** 2)))
        metrics.energy_mae_mev = metrics.energy_mae * 1000
    
    # Compute volume metrics
    if volume_errors:
        metrics.volume_mae_percent = float(np.mean(volume_errors))
    
    # Compute stability classification metrics
    if stability_pairs:
        tp = sum(1 for p, a in stability_pairs if p and a)
        fp = sum(1 for p, a in stability_pairs if p and not a)
        fn = sum(1 for p, a in stability_pairs if not p and a)
        tn = sum(1 for p, a in stability_pairs if not p and not a)
        
        metrics.stability_accuracy = (tp + tn) / len(stability_pairs)
        
        if tp + fp > 0:
            metrics.stability_precision = tp / (tp + fp)
        if tp + fn > 0:
            metrics.stability_recall = tp / (tp + fn)
        if metrics.stability_precision + metrics.stability_recall > 0:
            metrics.stability_f1 = (
                2 * metrics.stability_precision * metrics.stability_recall /
                (metrics.stability_precision + metrics.stability_recall)
            )
    
    return metrics


# =============================================================================
# SECTION 4: STABILITY ASSESSMENT METRICS
# =============================================================================

@dataclass
class StabilityMetrics:
    """
    Focused metrics for the stability assessment tool.
    
    Measures how well the system predicts thermodynamic stability
    and decomposition products.
    """
    # Hull distance accuracy
    e_hull_mae_mev: float = 0.0   # MAE in meV/atom
    e_hull_rmse_mev: float = 0.0  # RMSE in meV/atom
    
    # Classification accuracy
    verdict_accuracy: float = 0.0  # 4-class: stable/metastable/marginal/unstable
    binary_accuracy: float = 0.0   # 2-class: stable vs not
    
    # Decomposition prediction
    decomposition_accuracy: float = 0.0  # Correct decomposition products
    
    # Correlation
    pearson_r: float = 0.0  # Correlation between predicted and actual E_hull
    spearman_rho: float = 0.0  # Rank correlation
    
    def to_dict(self) -> Dict:
        return {
            "e_hull_mae_mev": round(self.e_hull_mae_mev, 2),
            "e_hull_rmse_mev": round(self.e_hull_rmse_mev, 2),
            "verdict_accuracy": round(self.verdict_accuracy, 4),
            "binary_accuracy": round(self.binary_accuracy, 4),
            "pearson_r": round(self.pearson_r, 4),
            "spearman_rho": round(self.spearman_rho, 4),
        }


def compute_stability_metrics(
    predicted_e_hulls: List[float],  # meV/atom
    actual_e_hulls: List[float],     # meV/atom
) -> StabilityMetrics:
    """
    Compute stability assessment metrics.
    
    Args:
        predicted_e_hulls: List of predicted E_above_hull values (meV)
        actual_e_hulls: List of DFT E_above_hull values (meV)
    
    Returns:
        StabilityMetrics
    """
    from scipy import stats
    
    metrics = StabilityMetrics()
    
    pred = np.array(predicted_e_hulls)
    actual = np.array(actual_e_hulls)
    
    # MAE and RMSE
    errors = pred - actual
    metrics.e_hull_mae_mev = float(np.mean(np.abs(errors)))
    metrics.e_hull_rmse_mev = float(np.sqrt(np.mean(errors ** 2)))
    
    # Classification accuracy
    def classify_4(e_hull):
        if e_hull < 25:
            return "stable"
        elif e_hull < 50:
            return "metastable"
        elif e_hull < 100:
            return "marginal"
        return "unstable"
    
    def classify_2(e_hull):
        return e_hull < 25
    
    pred_4 = [classify_4(e) for e in pred]
    actual_4 = [classify_4(e) for e in actual]
    metrics.verdict_accuracy = sum(p == a for p, a in zip(pred_4, actual_4)) / len(pred)
    
    pred_2 = [classify_2(e) for e in pred]
    actual_2 = [classify_2(e) for e in actual]
    metrics.binary_accuracy = sum(p == a for p, a in zip(pred_2, actual_2)) / len(pred)
    
    # Correlation
    if len(pred) > 2:
        metrics.pearson_r = float(stats.pearsonr(pred, actual)[0])
        metrics.spearman_rho = float(stats.spearmanr(pred, actual)[0])
    
    return metrics


# =============================================================================
# SECTION 5: AGENT-LEVEL METRICS
# =============================================================================

@dataclass
class AgentMetrics:
    """
    Metrics for evaluating the LLM agent's reasoning and planning.
    
    These metrics assess the quality of the agent's decisions,
    not just the final outputs.
    """
    # Task completion
    task_completion_rate: float = 0.0
    partial_completion_rate: float = 0.0
    
    # Tool usage
    tool_selection_accuracy: float = 0.0  # Did it pick the right tools?
    tool_usage_efficiency: float = 0.0    # Minimal tool calls to complete task
    avg_tool_calls_per_task: float = 0.0
    
    # Reasoning quality (LLM-as-Judge scores, 1-5 scale)
    reasoning_coherence: float = 0.0
    scientific_accuracy: float = 0.0
    appropriate_uncertainty: float = 0.0
    
    # Efficiency
    avg_execution_time_seconds: float = 0.0
    avg_tokens_per_task: int = 0
    
    # Error analysis
    error_rate: float = 0.0
    error_types: Dict[str, int] = field(default_factory=dict)
    
    # Detailed breakdown
    by_task_type: Dict[str, Dict] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "task_completion_rate": round(self.task_completion_rate, 4),
            "tool_selection_accuracy": round(self.tool_selection_accuracy, 4),
            "tool_usage_efficiency": round(self.tool_usage_efficiency, 4),
            "avg_tool_calls": round(self.avg_tool_calls_per_task, 2),
            "reasoning_coherence": round(self.reasoning_coherence, 2),
            "scientific_accuracy": round(self.scientific_accuracy, 2),
            "avg_execution_time_s": round(self.avg_execution_time_seconds, 2),
            "error_rate": round(self.error_rate, 4),
        }


def compute_agent_metrics(
    traces: List[AgentTrace],
    expected_tool_sequences: Optional[Dict[str, List[str]]] = None,
    llm_judge_scores: Optional[Dict[str, Dict[str, float]]] = None,
) -> AgentMetrics:
    """
    Compute agent-level metrics from execution traces.
    
    Args:
        traces: List of agent execution traces
        expected_tool_sequences: Optional mapping of task_id to expected tool sequence
        llm_judge_scores: Optional LLM-as-Judge scores for reasoning quality
    
    Returns:
        AgentMetrics
    """
    metrics = AgentMetrics()
    
    if not traces:
        return metrics
    
    # Task completion
    successful = sum(1 for t in traces if t.success)
    metrics.task_completion_rate = successful / len(traces)
    
    # Tool usage
    total_tool_calls = sum(len(t.tool_calls) for t in traces)
    metrics.avg_tool_calls_per_task = total_tool_calls / len(traces)
    
    # Tool selection accuracy
    if expected_tool_sequences:
        correct_selections = 0
        total_comparisons = 0
        for trace in traces:
            if trace.task_id in expected_tool_sequences:
                expected = set(expected_tool_sequences[trace.task_id])
                actual = set(call.get("tool_name", "") for call in trace.tool_calls)
                
                # Check if all expected tools were used
                if expected.issubset(actual):
                    correct_selections += 1
                total_comparisons += 1
        
        if total_comparisons > 0:
            metrics.tool_selection_accuracy = correct_selections / total_comparisons
    
    # Efficiency (fewer tool calls = better)
    if expected_tool_sequences:
        efficiency_scores = []
        for trace in traces:
            if trace.task_id in expected_tool_sequences:
                expected_count = len(expected_tool_sequences[trace.task_id])
                actual_count = len(trace.tool_calls)
                efficiency = min(1.0, expected_count / max(1, actual_count))
                efficiency_scores.append(efficiency)
        
        if efficiency_scores:
            metrics.tool_usage_efficiency = np.mean(efficiency_scores)
    
    # Timing and tokens
    metrics.avg_execution_time_seconds = np.mean([t.execution_time_seconds for t in traces])
    metrics.avg_tokens_per_task = int(np.mean([t.token_count for t in traces]))
    
    # Error analysis
    failed = [t for t in traces if not t.success]
    metrics.error_rate = len(failed) / len(traces)
    
    # LLM-as-Judge scores
    if llm_judge_scores:
        coherence_scores = []
        accuracy_scores = []
        uncertainty_scores = []
        
        for trace in traces:
            if trace.task_id in llm_judge_scores:
                scores = llm_judge_scores[trace.task_id]
                if "coherence" in scores:
                    coherence_scores.append(scores["coherence"])
                if "scientific_accuracy" in scores:
                    accuracy_scores.append(scores["scientific_accuracy"])
                if "uncertainty" in scores:
                    uncertainty_scores.append(scores["uncertainty"])
        
        if coherence_scores:
            metrics.reasoning_coherence = np.mean(coherence_scores)
        if accuracy_scores:
            metrics.scientific_accuracy = np.mean(accuracy_scores)
        if uncertainty_scores:
            metrics.appropriate_uncertainty = np.mean(uncertainty_scores)
    
    return metrics


# =============================================================================
# SECTION 6: UNIFIED METRICS COMPUTATION
# =============================================================================

@dataclass
class AllMetrics:
    """Container for all evaluation metrics."""
    screening: Optional[ScreeningMetrics] = None
    innovation: Optional[InnovationMetrics] = None
    stability: Optional[StabilityMetrics] = None
    agent: Optional[AgentMetrics] = None
    
    def to_dict(self) -> Dict:
        result = {}
        if self.screening:
            result["screening"] = self.screening.to_dict()
        if self.innovation:
            result["innovation"] = self.innovation.to_dict()
        if self.stability:
            result["stability"] = self.stability.to_dict()
        if self.agent:
            result["agent"] = self.agent.to_dict()
        return result
    
    def to_latex_table(self) -> str:
        """Generate LaTeX table for research paper."""
        lines = [
            r"\begin{table}[h]",
            r"\centering",
            r"\caption{Evaluation Metrics for Materials Project Deep Research}",
            r"\label{tab:evaluation_metrics}",
            r"\begin{tabular}{llr}",
            r"\toprule",
            r"\textbf{Category} & \textbf{Metric} & \textbf{Value} \\",
            r"\midrule",
        ]
        
        if self.screening:
            lines.append(r"\multirow{4}{*}{Screening} ")
            lines.append(f"& Precision & {self.screening.precision:.3f} \\\\")
            lines.append(f"& Recall & {self.screening.recall:.3f} \\\\")
            lines.append(f"& F1 Score & {self.screening.f1_score:.3f} \\\\")
            lines.append(f"& NDCG@10 & {self.screening.ndcg_at_10:.3f} \\\\")
            lines.append(r"\midrule")
        
        if self.innovation:
            lines.append(r"\multirow{4}{*}{Innovation} ")
            lines.append(f"& Energy MAE (meV) & {self.innovation.energy_mae_mev:.1f} \\\\")
            lines.append(f"& Stability F1 & {self.innovation.stability_f1:.3f} \\\\")
            lines.append(f"& Volume MAE (\\%) & {self.innovation.volume_mae_percent:.1f} \\\\")
            lines.append(f"& Success Rate & {self.innovation.confirmed_stable_count / max(1, self.innovation.predicted_stable_count):.3f} \\\\")
            lines.append(r"\midrule")
        
        if self.agent:
            lines.append(r"\multirow{3}{*}{Agent} ")
            lines.append(f"& Task Completion & {self.agent.task_completion_rate:.3f} \\\\")
            lines.append(f"& Tool Accuracy & {self.agent.tool_selection_accuracy:.3f} \\\\")
            lines.append(f"& Reasoning Score & {self.agent.reasoning_coherence:.2f}/5 \\\\")
        
        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ])
        
        return "\n".join(lines)


def compute_all_metrics(
    screening_predictions: Optional[List[MaterialPrediction]] = None,
    screening_ground_truth: Optional[List[MaterialGroundTruth]] = None,
    innovation_predictions: Optional[List[InnovationPrediction]] = None,
    innovation_ground_truth: Optional[List[InnovationGroundTruth]] = None,
    stability_predicted: Optional[List[float]] = None,
    stability_actual: Optional[List[float]] = None,
    agent_traces: Optional[List[AgentTrace]] = None,
    expected_tool_sequences: Optional[Dict[str, List[str]]] = None,
    llm_judge_scores: Optional[Dict[str, Dict[str, float]]] = None,
) -> AllMetrics:
    """
    Compute all metrics from provided data.
    
    Pass None for any category you don't want to evaluate.
    """
    result = AllMetrics()
    
    if screening_predictions and screening_ground_truth:
        result.screening = compute_screening_metrics(
            screening_predictions, screening_ground_truth
        )
    
    if innovation_predictions and innovation_ground_truth:
        result.innovation = compute_innovation_metrics(
            innovation_predictions, innovation_ground_truth
        )
    
    if stability_predicted and stability_actual:
        result.stability = compute_stability_metrics(
            stability_predicted, stability_actual
        )
    
    if agent_traces:
        result.agent = compute_agent_metrics(
            agent_traces, expected_tool_sequences, llm_judge_scores
        )
    
    return result
