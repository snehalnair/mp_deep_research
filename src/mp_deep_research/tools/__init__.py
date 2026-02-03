"""
Tools package for MP Deep Research Agent.

Imports are lazy to avoid pulling optional dependencies at import time.
"""
# src/mp_deep_research/tools/__init__.py

from .screening import batch_screen_materials
from .analysis import analyze_candidate
from .comparison import compare_structures, correlate_properties
from .plotting import generate_phase_diagram
from .utility import think
# NEW TOOLS
from .innovation import substitute_species, relax_structure_m3gnet, assess_stability
from .search_arxiv import search_arxiv
from langchain_core.tools import tool

ALL_TOOLS = [
    batch_screen_materials,
    analyze_candidate,
    compare_structures,
    generate_phase_diagram,
    correlate_properties,
    # identify_data_gaps,
    think,
    substitute_species,
    relax_structure_m3gnet,
    assess_stability,
    search_arxiv
]