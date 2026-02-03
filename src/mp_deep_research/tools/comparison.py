"""
Comparison tools for analyzing relationships between materials.
Handles structure matching and property correlation analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from langchain_core.tools import tool

# Scientific imports
from mp_api.client import MPRester
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from .shared import OUTPUT_DIR, load_from_cache, ensure_structure, get_structure_from_mp

plt.switch_backend("Agg")  # Non-interactive backend

# =============================================================================
# TOOL: STRUCTURE COMPARISON
# =============================================================================

@tool
def compare_structures(material_id_1: str, material_id_2: str) -> str:
    """
    Compare key properties of two materials side-by-side.
    """
    # --- CIRCUIT BREAKER ---
    if material_id_1 == material_id_2:
        return (
            f"ERROR: You are trying to compare {material_id_1} to itself. "
            f"This is redundant. If you want to analyze a single material, "
            f"use the 'analyze_candidate' tool instead."
        )
    # -----------------------

    try:
        with MPRester() as mpr:
            docs = mpr.materials.summary.search(material_ids=[material_id_1, material_id_2])
            
            # (Map results to IDs for safety)
            data = {str(d.material_id): d for d in docs}
            
            if material_id_1 not in data or material_id_2 not in data:
                return f"Error: Could not find one or both IDs: {material_id_1}, {material_id_2}"

            d1 = data[material_id_1]
            d2 = data[material_id_2]

            return f"""
STRUCTURE COMPARISON
{'='*50}
Material 1: {material_id_1}
  Formula: {d1.formula_pretty}
  Space group: {d1.symmetry.symbol} (#{d1.symmetry.number})
  Crystal system: {d1.symmetry.crystal_system}
  Volume: {d1.volume:.2f} Å³
  Band Gap: {d1.band_gap:.2f} eV
  Stability: {d1.energy_above_hull:.3f} eV/atom

Material 2: {material_id_2}
  Formula: {d2.formula_pretty}
  Space group: {d2.symmetry.symbol} (#{d2.symmetry.number})
  Crystal system: {d2.symmetry.crystal_system}
  Volume: {d2.volume:.2f} Å³
  Band Gap: {d2.band_gap:.2f} eV
  Stability: {d2.energy_above_hull:.3f} eV/atom
"""
    except Exception as e:
        return f"Comparison failed: {e}"


# =============================================================================
# TOOL: PROPERTY CORRELATION
# =============================================================================

@tool
def correlate_properties(
    chemsys: str,
    property_x: str = "band_gap",
    property_y: str = "energy_above_hull"
) -> str:
    """
    Analyze correlation between two properties across screened materials.
    
    Loads cached screening results and generates a scatter plot showing
    the relationship between two properties. Computes Pearson correlation.
    
    Args:
        chemsys: Chemical system (must have been screened previously)
        property_x: Property for x-axis (band_gap, formation_energy, volume, etc.)
        property_y: Property for y-axis
    
    Returns:
        Correlation analysis and path to scatter plot
    """
    # Load cached results
    result = load_from_cache(chemsys)
    
    if not result:
        return f"No cached screening results for {chemsys}. Run batch_screen_materials first."
    
    # Extract property values
    x_vals = []
    y_vals = []
    labels = []
    
    for c in result.candidates:
        x = getattr(c, property_x, None)
        y = getattr(c, property_y, None)
        if x is not None and y is not None:
            x_vals.append(x)
            y_vals.append(y)
            labels.append(c.material_id)
    
    if len(x_vals) < 3:
        return f"Insufficient data points ({len(x_vals)}) for correlation analysis."
    
    # Compute correlation
    corr = np.corrcoef(x_vals, y_vals)[0, 1]
    
    # Generate plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.scatter(x_vals, y_vals, c='steelblue', s=60, alpha=0.7, edgecolors='white')
    
    # Add trend line
    z = np.polyfit(x_vals, y_vals, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(x_vals), max(x_vals), 100)
    ax.plot(x_line, p(x_line), 'r--', alpha=0.7, label=f'Trend (r={corr:.2f})')
    
    ax.set_xlabel(property_x.replace('_', ' ').title())
    ax.set_ylabel(property_y.replace('_', ' ').title())
    ax.set_title(f"{property_x} vs {property_y} in {chemsys}")
    ax.legend()
    
    fig_path = OUTPUT_DIR / f"correlation_{chemsys.replace('-', '_')}_{property_x}_{property_y}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # Interpret correlation
    if abs(corr) > 0.7:
        strength = "Strong"
    elif abs(corr) > 0.4:
        strength = "Moderate"
    else:
        strength = "Weak"
    
    direction = "positive" if corr > 0 else "negative"
    
    summary = f"""
PROPERTY CORRELATION: {chemsys}
{'='*50}

Properties analyzed:
  X: {property_x}
  Y: {property_y}

Data points: {len(x_vals)}
Pearson correlation: {corr:.3f}
Interpretation: {strength} {direction} correlation

FIGURE SAVED:
  {fig_path}
"""
    
    return summary