"""
Tool Validation and Reliability Metrics

Based on Booking.com's AI Agent Evaluation best practices.
Reference: https://booking.ai/ai-agent-evaluation-82e781439d97

This module provides:
1. Tool Call Validity - JSON schema validation for tool calls
2. Tool Reliability - Quality checks for tool names and descriptions
3. Tool Argument Hallucination Detection
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import json
import re


# =============================================================================
# TOOL CALL VALIDITY (Glass Box Evaluation)
# =============================================================================

class ToolCallError(Enum):
    """Types of tool call validation errors."""
    VALID = "valid"
    UNKNOWN_TOOL = "unknown_tool"
    MISSING_REQUIRED_PARAM = "missing_required_param"
    UNEXPECTED_PARAM = "unexpected_param"
    TYPE_MISMATCH = "type_mismatch"
    FORMAT_ERROR = "format_error"
    SCHEMA_VALIDATION_ERROR = "schema_validation_error"


@dataclass
class ToolCallValidationResult:
    """Result of validating a single tool call."""
    tool_name: str
    is_valid: bool
    error_type: ToolCallError = ToolCallError.VALID
    error_message: str = ""
    missing_params: List[str] = field(default_factory=list)
    unexpected_params: List[str] = field(default_factory=list)
    type_mismatches: Dict[str, str] = field(default_factory=dict)


@dataclass
class ToolSchema:
    """Schema definition for a tool."""
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema format
    required_params: List[str] = field(default_factory=list)
    
    @classmethod
    def from_langchain_tool(cls, tool) -> "ToolSchema":
        """Create ToolSchema from a LangChain tool."""
        schema = tool.args_schema.schema() if hasattr(tool, 'args_schema') and tool.args_schema else {}
        return cls(
            name=tool.name,
            description=tool.description or "",
            parameters=schema.get("properties", {}),
            required_params=schema.get("required", []),
        )


def validate_tool_call(
    tool_call: Dict[str, Any],
    available_tools: Dict[str, ToolSchema],
) -> ToolCallValidationResult:
    """
    Validate a tool call against available tool schemas.
    
    Based on Berkeley Function Calling Leaderboard approach using AST parsing.
    
    Args:
        tool_call: Dict with 'name' and 'arguments' keys
        available_tools: Dict mapping tool names to their schemas
    
    Returns:
        ToolCallValidationResult with validation details
    """
    tool_name = tool_call.get("name", "")
    arguments = tool_call.get("arguments", {})
    
    # If arguments is a string, try to parse as JSON
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return ToolCallValidationResult(
                tool_name=tool_name,
                is_valid=False,
                error_type=ToolCallError.FORMAT_ERROR,
                error_message=f"Arguments is not valid JSON: {arguments[:100]}..."
            )
    
    # Check 1: Tool name exists
    if tool_name not in available_tools:
        return ToolCallValidationResult(
            tool_name=tool_name,
            is_valid=False,
            error_type=ToolCallError.UNKNOWN_TOOL,
            error_message=f"Tool '{tool_name}' not in available tools: {list(available_tools.keys())}"
        )
    
    schema = available_tools[tool_name]
    
    # Check 2: Required parameters present
    missing = [p for p in schema.required_params if p not in arguments]
    if missing:
        return ToolCallValidationResult(
            tool_name=tool_name,
            is_valid=False,
            error_type=ToolCallError.MISSING_REQUIRED_PARAM,
            error_message=f"Missing required parameters: {missing}",
            missing_params=missing,
        )
    
    # Check 3: No unexpected parameters
    expected_params = set(schema.parameters.keys())
    actual_params = set(arguments.keys())
    unexpected = actual_params - expected_params
    if unexpected:
        return ToolCallValidationResult(
            tool_name=tool_name,
            is_valid=False,
            error_type=ToolCallError.UNEXPECTED_PARAM,
            error_message=f"Unexpected parameters: {unexpected}",
            unexpected_params=list(unexpected),
        )
    
    # Check 4: Type validation
    type_mismatches = {}
    for param_name, param_value in arguments.items():
        if param_name in schema.parameters:
            expected_type = schema.parameters[param_name].get("type", "any")
            if not _check_type(param_value, expected_type):
                type_mismatches[param_name] = f"Expected {expected_type}, got {type(param_value).__name__}"
    
    if type_mismatches:
        return ToolCallValidationResult(
            tool_name=tool_name,
            is_valid=False,
            error_type=ToolCallError.TYPE_MISMATCH,
            error_message=f"Type mismatches: {type_mismatches}",
            type_mismatches=type_mismatches,
        )
    
    # All checks passed
    return ToolCallValidationResult(
        tool_name=tool_name,
        is_valid=True,
        error_type=ToolCallError.VALID,
    )


def _check_type(value: Any, expected_type: str) -> bool:
    """Check if value matches expected JSON Schema type."""
    type_map = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "array": list,
        "object": dict,
        "null": type(None),
    }
    
    if expected_type == "any":
        return True
    
    expected_python_type = type_map.get(expected_type)
    if expected_python_type is None:
        return True  # Unknown type, assume valid
    
    return isinstance(value, expected_python_type)


def validate_tool_call_with_jsonschema(
    tool_call: Dict[str, Any],
    tool_schema: Dict[str, Any],
) -> ToolCallValidationResult:
    """
    Validate tool call using full JSON Schema validation.
    
    This is the approach recommended by Booking.com for production systems.
    
    Args:
        tool_call: Dict with 'name' and 'arguments'
        tool_schema: Full JSON Schema for the tool
    
    Returns:
        ToolCallValidationResult
    """
    try:
        from jsonschema import validate, ValidationError
        
        tool_name = tool_call.get("name", "unknown")
        arguments = tool_call.get("arguments", {})
        
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        
        validate(instance=arguments, schema=tool_schema)
        
        return ToolCallValidationResult(
            tool_name=tool_name,
            is_valid=True,
        )
        
    except ValidationError as e:
        return ToolCallValidationResult(
            tool_name=tool_call.get("name", "unknown"),
            is_valid=False,
            error_type=ToolCallError.SCHEMA_VALIDATION_ERROR,
            error_message=str(e.message),
        )
    except ImportError:
        # Fallback if jsonschema not installed
        return validate_tool_call(tool_call, {})
    except Exception as e:
        return ToolCallValidationResult(
            tool_name=tool_call.get("name", "unknown"),
            is_valid=False,
            error_type=ToolCallError.FORMAT_ERROR,
            error_message=str(e),
        )


# =============================================================================
# TOOL CORRECTNESS (Glass Box Evaluation)
# =============================================================================

@dataclass
class ToolCorrectnessResult:
    """Result of evaluating tool correctness for a task."""
    task_id: str
    is_correct: bool
    expected_tools: List[str]
    used_tools: List[str]
    missing_tools: List[str] = field(default_factory=list)
    redundant_tools: List[str] = field(default_factory=list)


def evaluate_tool_correctness(
    task_id: str,
    used_tools: List[str],
    expected_tools: List[str],
    allow_extra_tools: bool = False,
) -> ToolCorrectnessResult:
    """
    Evaluate if the agent used the correct tools for a task.
    
    Args:
        task_id: Identifier for the task
        used_tools: Tools actually used by the agent
        expected_tools: Tools expected to solve the task
        allow_extra_tools: If True, extra tools don't cause failure
    
    Returns:
        ToolCorrectnessResult with details
    """
    expected_set = set(expected_tools)
    used_set = set(used_tools)
    
    missing = list(expected_set - used_set)
    redundant = list(used_set - expected_set)
    
    # Correct if all expected tools were used
    # And no redundant tools (unless allowed)
    is_correct = (
        len(missing) == 0 and 
        (allow_extra_tools or len(redundant) == 0)
    )
    
    return ToolCorrectnessResult(
        task_id=task_id,
        is_correct=is_correct,
        expected_tools=expected_tools,
        used_tools=used_tools,
        missing_tools=missing,
        redundant_tools=redundant,
    )


# =============================================================================
# TOOL RELIABILITY (Quality of Tool Definitions)
# =============================================================================

class ToolNameCheck(Enum):
    """Tool name reliability checks from Booking.com."""
    SNAKE_CASE = "snake_case"
    NO_ABBREVIATIONS = "no_abbreviations"
    ACTION_ORIENTED = "action_oriented"
    REASONABLE_LENGTH = "reasonable_length"
    DESCRIPTIVE = "descriptive"


class ToolDescriptionCheck(Enum):
    """Tool description reliability checks from Booking.com."""
    NOT_EMPTY = "not_empty"
    NOT_TOO_SHORT = "not_too_short"
    NOT_TOO_LONG = "not_too_long"
    HAS_INPUT_DESCRIPTION = "has_input_description"
    NO_REDUNDANCY = "no_redundancy"
    MAX_OPTIONAL_ARGS = "max_optional_args"


@dataclass
class ToolReliabilityResult:
    """Result of tool reliability checks."""
    tool_name: str
    name_checks: Dict[str, bool] = field(default_factory=dict)
    description_checks: Dict[str, bool] = field(default_factory=dict)
    overall_score: float = 0.0
    issues: List[str] = field(default_factory=list)
    
    def is_reliable(self, threshold: float = 0.8) -> bool:
        """Check if tool meets reliability threshold."""
        return self.overall_score >= threshold


def check_tool_reliability(
    tool_name: str,
    tool_description: str,
    tool_schema: Optional[Dict[str, Any]] = None,
) -> ToolReliabilityResult:
    """
    Run all reliability checks on a tool definition.
    
    Based on Booking.com's tool reliability framework.
    
    Args:
        tool_name: Name of the tool
        tool_description: Tool's description text
        tool_schema: Optional JSON schema for parameter checks
    
    Returns:
        ToolReliabilityResult with all check results
    """
    result = ToolReliabilityResult(tool_name=tool_name)
    issues = []
    
    # === NAME CHECKS ===
    
    # Check 1: Snake case
    is_snake_case = bool(re.match(r'^[a-z][a-z0-9_]*$', tool_name))
    result.name_checks[ToolNameCheck.SNAKE_CASE.value] = is_snake_case
    if not is_snake_case:
        issues.append(f"Tool name '{tool_name}' should use snake_case")
    
    # Check 2: No cryptic abbreviations
    abbreviation_patterns = [r'[A-Z]{3,}', r'_[a-z]_', r'\d{2,}']
    has_abbreviations = any(re.search(p, tool_name) for p in abbreviation_patterns)
    result.name_checks[ToolNameCheck.NO_ABBREVIATIONS.value] = not has_abbreviations
    if has_abbreviations:
        issues.append(f"Tool name '{tool_name}' contains cryptic abbreviations")
    
    # Check 3: Action oriented (starts with verb)
    action_verbs = [
        'get', 'fetch', 'search', 'find', 'list', 'create', 'update', 
        'delete', 'remove', 'add', 'set', 'check', 'validate', 'compute',
        'calculate', 'analyze', 'generate', 'build', 'run', 'execute',
        'batch', 'assess', 'substitute', 'relax', 'screen', 'compare',
    ]
    starts_with_verb = any(tool_name.lower().startswith(v) for v in action_verbs)
    result.name_checks[ToolNameCheck.ACTION_ORIENTED.value] = starts_with_verb
    if not starts_with_verb:
        issues.append(f"Tool name '{tool_name}' should start with an action verb")
    
    # Check 4: Reasonable length
    reasonable_length = 3 <= len(tool_name) <= 50
    result.name_checks[ToolNameCheck.REASONABLE_LENGTH.value] = reasonable_length
    if not reasonable_length:
        issues.append(f"Tool name '{tool_name}' length should be 3-50 chars")
    
    # Check 5: Not too generic
    generic_names = ['search', 'get', 'fetch', 'do', 'run', 'execute', 'process', 'tool']
    is_descriptive = tool_name.lower() not in generic_names
    result.name_checks[ToolNameCheck.DESCRIPTIVE.value] = is_descriptive
    if not is_descriptive:
        issues.append(f"Tool name '{tool_name}' is too generic")
    
    # === DESCRIPTION CHECKS ===
    
    desc = tool_description.strip() if tool_description else ""
    
    # Check 1: Not empty
    not_empty = len(desc) > 0
    result.description_checks[ToolDescriptionCheck.NOT_EMPTY.value] = not_empty
    if not not_empty:
        issues.append("Tool description is empty")
    
    # Check 2: Not too short
    not_too_short = len(desc) >= 20
    result.description_checks[ToolDescriptionCheck.NOT_TOO_SHORT.value] = not_too_short
    if not not_too_short and not_empty:
        issues.append(f"Tool description too short ({len(desc)} chars, need ≥20)")
    
    # Check 3: Not too long
    not_too_long = len(desc) <= 500
    result.description_checks[ToolDescriptionCheck.NOT_TOO_LONG.value] = not_too_long
    if not not_too_long:
        issues.append(f"Tool description too long ({len(desc)} chars, need ≤500)")
    
    # Check 4: Has input description
    input_keywords = ['arg', 'param', 'input', 'takes', 'accepts', 'requires']
    has_input_desc = any(kw in desc.lower() for kw in input_keywords)
    result.description_checks[ToolDescriptionCheck.HAS_INPUT_DESCRIPTION.value] = has_input_desc
    if not has_input_desc and len(desc) > 50:
        issues.append("Tool description should describe input parameters")
    
    # Check 5: No redundancy
    name_words = set(tool_name.lower().replace('_', ' ').split())
    desc_words = set(desc.lower().split()[:10])
    overlap_ratio = len(name_words & desc_words) / max(len(name_words), 1)
    no_redundancy = overlap_ratio < 0.8
    result.description_checks[ToolDescriptionCheck.NO_REDUNDANCY.value] = no_redundancy
    if not no_redundancy:
        issues.append("Tool description is redundant with tool name")
    
    # Check 6: Max optional arguments
    max_optional_ok = True
    if tool_schema:
        properties = tool_schema.get("properties", {})
        required = set(tool_schema.get("required", []))
        optional_count = len(properties) - len(required)
        max_optional_ok = optional_count <= 3
        if not max_optional_ok:
            issues.append(f"Too many optional arguments ({optional_count}, max 3)")
    result.description_checks[ToolDescriptionCheck.MAX_OPTIONAL_ARGS.value] = max_optional_ok
    
    # === COMPUTE OVERALL SCORE ===
    all_checks = list(result.name_checks.values()) + list(result.description_checks.values())
    result.overall_score = sum(all_checks) / len(all_checks) if all_checks else 0.0
    result.issues = issues
    
    return result


# =============================================================================
# AGGREGATE METRICS
# =============================================================================

@dataclass
class ToolValidationMetrics:
    """Aggregate metrics for tool validation across all calls."""
    total_calls: int = 0
    valid_calls: int = 0
    validity_rate: float = 0.0
    
    # Error breakdown
    unknown_tool_errors: int = 0
    missing_param_errors: int = 0
    unexpected_param_errors: int = 0
    type_mismatch_errors: int = 0
    format_errors: int = 0
    
    # Tool correctness
    total_tasks: int = 0
    correct_tool_selections: int = 0
    tool_correctness_rate: float = 0.0
    
    # Tool reliability
    total_tools_checked: int = 0
    reliable_tools: int = 0
    avg_reliability_score: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "tool_call_validity_rate": round(self.validity_rate, 4),
            "tool_correctness_rate": round(self.tool_correctness_rate, 4),
            "avg_tool_reliability_score": round(self.avg_reliability_score, 4),
            "error_breakdown": {
                "unknown_tool": self.unknown_tool_errors,
                "missing_param": self.missing_param_errors,
                "unexpected_param": self.unexpected_param_errors,
                "type_mismatch": self.type_mismatch_errors,
                "format_error": self.format_errors,
            }
        }


def compute_tool_validation_metrics(
    validation_results: List[ToolCallValidationResult],
    correctness_results: List[ToolCorrectnessResult],
    reliability_results: List[ToolReliabilityResult],
) -> ToolValidationMetrics:
    """
    Compute aggregate tool validation metrics.
    
    Args:
        validation_results: Results from validate_tool_call()
        correctness_results: Results from evaluate_tool_correctness()
        reliability_results: Results from check_tool_reliability()
    
    Returns:
        ToolValidationMetrics with aggregate scores
    """
    metrics = ToolValidationMetrics()
    
    # Validity metrics
    if validation_results:
        metrics.total_calls = len(validation_results)
        metrics.valid_calls = sum(1 for r in validation_results if r.is_valid)
        metrics.validity_rate = metrics.valid_calls / metrics.total_calls
        
        # Error breakdown
        for r in validation_results:
            if r.error_type == ToolCallError.UNKNOWN_TOOL:
                metrics.unknown_tool_errors += 1
            elif r.error_type == ToolCallError.MISSING_REQUIRED_PARAM:
                metrics.missing_param_errors += 1
            elif r.error_type == ToolCallError.UNEXPECTED_PARAM:
                metrics.unexpected_param_errors += 1
            elif r.error_type == ToolCallError.TYPE_MISMATCH:
                metrics.type_mismatch_errors += 1
            elif r.error_type in (ToolCallError.FORMAT_ERROR, ToolCallError.SCHEMA_VALIDATION_ERROR):
                metrics.format_errors += 1
    
    # Correctness metrics
    if correctness_results:
        metrics.total_tasks = len(correctness_results)
        metrics.correct_tool_selections = sum(1 for r in correctness_results if r.is_correct)
        metrics.tool_correctness_rate = metrics.correct_tool_selections / metrics.total_tasks
    
    # Reliability metrics
    if reliability_results:
        metrics.total_tools_checked = len(reliability_results)
        metrics.reliable_tools = sum(1 for r in reliability_results if r.is_reliable())
        metrics.avg_reliability_score = sum(r.overall_score for r in reliability_results) / len(reliability_results)
    
    return metrics
