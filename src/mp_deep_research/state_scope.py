
"""State Definitions and Pydantic Schemas for Materials Project Research Scoping.

This defines the state objects and structured schemas used for
the materials research agent scoping workflow.
"""

import operator
from typing_extensions import Annotated, Sequence, Optional, List, Literal

from langchain_core.messages import BaseMessage
from langgraph.graph import MessagesState
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field, ConfigDict

# ===== MATERIALS-SPECIFIC ENUMS =====

DFTLevel = Literal["GGA", "GGA+U", "r2SCAN", "HSE06", "any"]
ResearchType = Literal["screening", "discovery", "property_calculation", "structure_prediction", "comparison"]

# ===== STATE DEFINITIONS =====

class PropertyThreshold(BaseModel):
    """Schema for numeric property constraints."""

    model_config = ConfigDict(extra="forbid")

    property_name: str = Field(
        description="Property name in MP schema (e.g., band_gap, energy_above_hull)."
    )
    min_value: Optional[float] = Field(
        default=None,
        description="Minimum allowed value for the property (inclusive)."
    )
    max_value: Optional[float] = Field(
        default=None,
        description="Maximum allowed value for the property (inclusive)."
    )
    unit: Optional[str] = Field(
        default=None,
        description="Unit for the property values (e.g., eV, eV/atom)."
    )

class AgentInputState(MessagesState):
    """Input state for the full agent - only contains messages from user input."""
    pass


class MaterialsResearchState(MessagesState):
    """
    Main state for the Materials Project multi-agent research system.

    Extends MessagesState with materials-specific fields for research coordination.
    """

    # ===== SCOPING FIELDS =====
    # Research brief generated from user conversation
    research_brief: Optional[str] = None

    # ===== MATERIALS-SPECIFIC FIELDS =====
    # Chemical system being studied (e.g., "Li-Co-O", ["Li", "Fe", "P", "O"])
    chemical_system: Optional[str] = None
    # Target properties to investigate
    target_properties: List[str] = []
    # Stability threshold (e_above_hull in eV/atom)
    stability_threshold: Optional[float] = 0.025  # Default: 25 meV/atom
    # DFT level of theory preference
    dft_level: DFTLevel = "any"
    # Type of research being conducted
    research_type: Optional[ResearchType] = None
    # Quantitative thresholds specified by user
    property_thresholds: List[PropertyThreshold] = []

    # ===== COORDINATION FIELDS =====
    # Messages exchanged with the supervisor agent
    supervisor_messages: Annotated[Sequence[BaseMessage], add_messages] = []

    # ===== RESEARCH FIELDS (used in later phases) =====
    # Material IDs found during research
    material_ids: List[str] = []
    # Raw unprocessed research notes
    raw_notes: Annotated[List[str], operator.add] = []
    # Processed notes ready for report generation
    notes: Annotated[List[str], operator.add] = []
    # Final research report
    final_report: Optional[str] = None
    # Generated visualization paths
    visualizations: List[str] = []
    # Generated workflow code (for simulation phase)
    workflow_code: Optional[str] = None

# Alias for backward compatibility
AgentState = MaterialsResearchState

# ===== STRUCTURED OUTPUT SCHEMAS =====

class ClarifyWithUser(BaseModel):
    """Schema for user clarification decision and questions - Materials Project specific."""

    model_config = ConfigDict(extra="forbid")

    need_clarification: bool = Field(
        description="Whether the user needs to be asked a clarifying question about their materials research request.",
    )
    question: str = Field(
        description="A question to clarify the materials research scope (chemical system, properties, thresholds, etc.)",
    )
    verification: str = Field(
        description="Verification message confirming the research scope and approach.",
    )
    # Materials-specific extracted information
    detected_chemical_system: Optional[str] = Field(
        default=None,
        description="Chemical system detected from user input (e.g., 'Li-Co-O')"
    )
    detected_properties: List[str] = Field(
        default=[],
        description="Properties detected from user input (e.g., ['band_gap', 'formation_energy'])"
    )
    detected_stability_threshold: Optional[float] = Field(
        default=None,
        description="Stability threshold detected from user input (e_above_hull in eV/atom)"
    )
    detected_dft_level: Optional[DFTLevel] = Field(
        default=None,
        description="DFT level of theory preference detected from user input"
    )

class ResearchQuestion(BaseModel):
    """Schema for structured research brief generation - Materials Project specific."""

    model_config = ConfigDict(extra="forbid")

    research_brief: str = Field(
        description="A detailed research question optimized for Materials Project database queries.",
    )
    chemical_system: Optional[str] = Field(
        default=None,
        description="The chemical system to investigate (e.g., 'Li-Fe-P-O')"
    )
    target_properties: List[str] = Field(
        default=[],
        description="List of properties to query from MP (e.g., ['band_gap', 'formation_energy_per_atom'])"
    )
    property_thresholds: List[PropertyThreshold] = Field(
        default_factory=list,
        description="Quantitative thresholds for properties with min/max bounds."
    )
    stability_threshold: Optional[float] = Field(
        default=0.025,
        description="Stability threshold (energy_above_hull) in eV/atom"
    )
    dft_level: DFTLevel = Field(
        default="any",
        description="Preferred DFT level of theory (GGA, GGA+U, r2SCAN, HSE06, any)"
    )
    research_type: str = Field(
        default="screening",
        description="Type of research: screening, discovery, property_calculation, structure_prediction, comparison"
    )
    requires_simulation: bool = Field(
        default=False,
        description="Whether this research will require DFT simulation workflows"
    )
    suggested_mp_query: Optional[str] = Field(
        default=None,
        description="Suggested MP API query structure for this research"
    )
