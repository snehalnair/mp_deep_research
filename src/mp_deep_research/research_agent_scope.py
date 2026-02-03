
"""User Clarification and Research Brief Generation for Materials Project.

This module implements the scoping phase of the materials research workflow:
1. Assess if the user's request needs clarification (chemical system, properties, thresholds)
2. Validate MP-specific requirements
3. Generate a detailed research brief optimized for MP queries
"""

from datetime import datetime
from typing import Literal
import os

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, get_buffer_string
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

# Import from our modules
try:
    from mp_deep_research.prompts import (
        clarify_with_user_instructions, 
        transform_messages_into_research_topic_prompt
    )
    from mp_deep_research.state_scope import (
        MaterialsResearchState, 
        ClarifyWithUser, 
        ResearchQuestion, 
        AgentInputState
    )
except ImportError:
    # For notebook execution
    from materials_project_prompts_v3_final import clarify_with_user_instructions, transform_messages_into_research_topic_prompt
    from state_scope import MaterialsResearchState, ClarifyWithUser, ResearchQuestion, AgentInputState

# ===== UTILITY FUNCTIONS =====

def get_today_str() -> str:
    """Get current date in a human-readable format."""
    return datetime.now().strftime("%a %b %d, %Y")

def validate_chemical_system(system: str) -> bool:
    """Validate that a chemical system string is properly formatted for MP."""
    if not system:
        return False
    # MP format: Element1-Element2-Element3 (e.g., "Li-Co-O")
    elements = system.split("-")
    # Basic validation: each part should be 1-2 characters, capitalized
    return all(len(e) <= 2 and e[0].isupper() for e in elements)

# ===== CONFIGURATION =====

# Initialize model - configurable via environment variable
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")
model = init_chat_model(model=f"openai:{LLM_MODEL}", temperature=0.0)

# ===== WORKFLOW NODES =====

def clarify_with_user(state: MaterialsResearchState) -> Command[Literal["write_research_brief", "__end__"]]:
    """
    Determine if the user's materials research request needs clarification.

    Materials-specific checks:
    - Chemical system specified?
    - Properties quantified (not vague terms like 'good')?
    - Stability requirements clear?
    """
    # Set up structured output model
    structured_output_model = model.with_structured_output(ClarifyWithUser)

    # Invoke the model with materials-specific clarification instructions
    response = structured_output_model.invoke([
        HumanMessage(content=clarify_with_user_instructions.format(
            messages=get_buffer_string(messages=state["messages"]), 
            date=get_today_str()
        ))
    ])

    # Route based on clarification need
    if response.need_clarification:
        return Command(
            goto=END, 
            update={
                "messages": [AIMessage(content=response.question)],
                # Store any detected information
                "chemical_system": response.detected_chemical_system,
                "target_properties": response.detected_properties or []
            }
        )
    else:
        return Command(
            goto="write_research_brief", 
            update={
                "messages": [AIMessage(content=response.verification)],
                "chemical_system": response.detected_chemical_system,
                "target_properties": response.detected_properties or []
            }
        )

def write_research_brief(state: MaterialsResearchState):
    """
    Transform conversation into a Materials Project-optimized research brief.

    Outputs:
    - Structured research question
    - Chemical system in MP format
    - Target properties list
    - Property thresholds
    - Suggested MP API query
    """
    # Set up structured output model
    structured_output_model = model.with_structured_output(ResearchQuestion)

    # Generate research brief from conversation history
    response = structured_output_model.invoke([
        HumanMessage(content=transform_messages_into_research_topic_prompt.format(
            messages=get_buffer_string(state.get("messages", [])),
            date=get_today_str()
        ))
    ])

    # Validate chemical system format
    chemical_system = response.chemical_system
    if chemical_system and not validate_chemical_system(chemical_system):
        # Try to fix common issues
        chemical_system = chemical_system.replace(" ", "-").replace(",", "-")

    # Update state with generated research brief and MP-specific fields
    return {
        "research_brief": response.research_brief,
        "chemical_system": chemical_system or state.get("chemical_system"),
        "target_properties": response.target_properties or state.get("target_properties", []),
        "property_thresholds": response.property_thresholds,
        "research_type": response.research_type,
        "supervisor_messages": [HumanMessage(content=f"{response.research_brief}.")]
    }

# ===== GRAPH CONSTRUCTION =====

def build_scoping_workflow():
    """Build and compile the scoping workflow."""
    # Build the scoping workflow
    builder = StateGraph(MaterialsResearchState, input_schema=AgentInputState)

    # Add workflow nodes
    builder.add_node("clarify_with_user", clarify_with_user)
    builder.add_node("write_research_brief", write_research_brief)

    # Add workflow edges
    builder.add_edge(START, "clarify_with_user")
    builder.add_edge("write_research_brief", END)

    # Compile the workflow
    return builder.compile()

# Create the compiled workflow
scope_research = build_scoping_workflow()
