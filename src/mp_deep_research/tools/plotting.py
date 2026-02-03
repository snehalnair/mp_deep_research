"""
Plotting tools for generating visual summaries of materials data.
Handles phase diagram generation and visualization.
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from langchain_core.tools import tool

# Scientific imports
from mp_api.client import MPRester
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDPlotter

from .shared import OUTPUT_DIR, ensure_structure

# =============================================================================
# TOOL: PHASE DIAGRAM
# =============================================================================

@tool(parse_docstring=True)
def generate_phase_diagram(chemsys: str, show_unstable: float = 0.05) -> str:
    """
    Generate a phase diagram for a chemical system.
    
    Creates a publication-quality phase diagram showing stable phases
    and optionally metastable phases up to a specified energy above hull.
    
    Args:
        chemsys: Chemical system (e.g., "Li-Fe-P-O", "Bi-Te")
        show_unstable: Show unstable phases up to this energy above hull (eV/atom)
    
    Returns:
        Summary of stable phases and path to generated figure
    """
    elements = chemsys.split("-")
    
    try:
        with MPRester() as mpr:
            entries = mpr.get_entries_in_chemsys(elements)
    except Exception as e:
        return f"Error retrieving phase diagram entries: {e}"
    
    if not entries:
        return f"No entries found for chemical system {chemsys}"
    
    try:
        pd = PhaseDiagram(entries)
    except Exception as e:
        return f"Error building phase diagram: {e}"
    
    # Get stable phases
    stable_entries = pd.stable_entries
    stable_phases = []
    for entry in stable_entries:
        stable_phases.append({
            'name': entry.name,
            'composition': entry.composition.reduced_formula,
            'energy': entry.energy_per_atom
        })
    
    # Generate plot
    try:
        plotter = PDPlotter(pd, show_unstable=show_unstable)
        fig = plotter.get_plot()
        plt.title(f"Phase Diagram: {chemsys}")
        
        fig_path = OUTPUT_DIR / f"phase_diagram_{chemsys.replace('-', '_')}.png"
        plt.savefig(fig_path, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()
    except Exception as e:
        return f"Error generating phase diagram plot: {e}"
    
    # Format summary
    summary = f"""
PHASE DIAGRAM: {chemsys}
{'='*50}

Total entries: {len(entries)}
Stable phases: {len(stable_entries)}

STABLE PHASES:
"""
    
    for phase in sorted(stable_phases, key=lambda x: x['energy']):
        summary += f"  • {phase['composition']}: {phase['energy']:.3f} eV/atom\n"
    
    summary += f"""
FIGURE SAVED:
  {fig_path}

NOTE: Metastable phases (up to {show_unstable*1000:.0f} meV above hull) shown in lighter color.
"""
    
    return summary