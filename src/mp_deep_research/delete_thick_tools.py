
"""
Thick Tools for Materials Research Agent.

Each tool encapsulates a complete workflow:
- Heavy computation happens in Python
- Rich objects (Structures, DOS) stay in Python
- Agent receives human-readable summaries
"""

import os
import pickle
from pathlib import Path
from typing import Optional, List, Tuple
from collections import Counter

from langchain_core.tools import tool

# MP and pymatgen imports
from mp_api.client import MPRester
from pymatgen.core import Structure
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDPlotter
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.electronic_structure.plotter import BSPlotter, DosPlotter
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

from mp_deep_research.models import MaterialCandidate, ScreeningResult, DataSource

# Directories
CACHE_DIR = Path("./cache")
OUTPUT_DIR = Path("./outputs")
CACHE_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


# =============================================================================
# TOOL 1: BATCH SCREENING
# =============================================================================

@tool(parse_docstring=True)
def batch_screen_materials(
    chemsys: str,
    stability_threshold: float = 0.025,
    band_gap_min: Optional[float] = None,
    band_gap_max: Optional[float] = None,
    require_semiconductor: bool = False,
) -> str:
    """
    Screen all materials in a chemical system against specified criteria.

    This tool queries the Materials Project for ALL materials in the given
    chemical system, applies filters, ranks candidates, and returns a summary.
    Full results are cached for subsequent analysis.

    Args:
        chemsys: Chemical system in MP format (e.g., "Li-Fe-P-O", "Bi-Te")
        stability_threshold: Max energy above hull in eV/atom (default 0.025)
        band_gap_min: Minimum band gap in eV (optional)
        band_gap_max: Maximum band gap in eV (optional)
        require_semiconductor: If True, exclude metals (optional)

    Returns:
        Summary of screening results including top candidates and exclusion statistics
    """
    # ===== QUERY PHASE =====
    with MPRester() as mpr:
        docs = mpr.materials.summary.search(
            chemsys=chemsys,
            fields=[
                "material_id", "formula_pretty", "composition",
                "band_gap", "is_metal", "is_gap_direct",
                "formation_energy_per_atom", "energy_above_hull", "is_stable",
                "symmetry", "volume", "density", "nsites", "theoretical"
            ]
        )

    if not docs:
        return f"No materials found in chemical system {chemsys}."

    # ===== CONVERT TO CANDIDATES =====
    all_candidates = []
    for doc in docs:
        candidate = MaterialCandidate(
            material_id=str(doc.material_id),
            formula=doc.formula_pretty,
            formation_energy=doc.formation_energy_per_atom,
            energy_above_hull=doc.energy_above_hull,
            is_stable=doc.is_stable,
            band_gap=doc.band_gap,
            is_metal=doc.is_metal,
            is_direct_gap=doc.is_gap_direct,
            space_group=doc.symmetry.symbol if doc.symmetry else None,
            space_group_number=doc.symmetry.number if doc.symmetry else None,
            crystal_system=doc.symmetry.crystal_system.value if doc.symmetry else None,
            volume=doc.volume,
            density=doc.density,
            nsites=doc.nsites,
            structure_source="ICSD" if not doc.theoretical else "DFT-predicted",
            is_theoretical=doc.theoretical,
        )
        all_candidates.append(candidate)

    total_queried = len(all_candidates)

    # ===== SCREENING PHASE =====
    exclusion_reasons = Counter()
    passed = []

    for c in all_candidates:
        # Stability filter
        if c.energy_above_hull is None:
            c.exclusion_reason = "missing_stability_data"
            exclusion_reasons["missing_stability_data"] += 1
            continue
        if c.energy_above_hull > stability_threshold:
            c.exclusion_reason = f"unstable (>{stability_threshold*1000:.0f} meV)"
            exclusion_reasons["unstable"] += 1
            continue

        # Semiconductor filter
        if require_semiconductor and c.is_metal:
            c.exclusion_reason = "metallic"
            exclusion_reasons["metallic"] += 1
            continue

        # Band gap filter
        if band_gap_min is not None or band_gap_max is not None:
            if c.band_gap is None:
                c.exclusion_reason = "missing_bandgap_data"
                exclusion_reasons["missing_bandgap_data"] += 1
                continue
            if band_gap_min is not None and c.band_gap < band_gap_min:
                c.exclusion_reason = f"bandgap_too_small (<{band_gap_min} eV)"
                exclusion_reasons["bandgap_too_small"] += 1
                continue
            if band_gap_max is not None and c.band_gap > band_gap_max:
                c.exclusion_reason = f"bandgap_too_large (>{band_gap_max} eV)"
                exclusion_reasons["bandgap_too_large"] += 1
                continue

        passed.append(c)

    # ===== RANKING PHASE =====
    for c in passed:
        score = 0.0
        # Stability score (lower e_hull = higher score)
        if c.energy_above_hull is not None:
            score += 0.4 * max(0, 1 - c.energy_above_hull / 0.05)
        # Provenance score (ICSD > predicted)
        score += 0.3 if not c.is_theoretical else 0.15
        # Has band gap data
        if c.band_gap is not None:
            score += 0.3
        c.rank_score = score

    ranked = sorted(passed, key=lambda x: x.rank_score, reverse=True)

    # ===== CACHE RESULTS =====
    cache_path = CACHE_DIR / f"{chemsys.replace('-', '_')}_screening.pkl"
    result = ScreeningResult(
        chemsys=chemsys,
        total_queried=total_queried,
        total_passed=len(ranked),
        candidates=ranked,
        exclusion_stats=dict(exclusion_reasons),
        cache_path=str(cache_path)
    )
    with open(cache_path, 'wb') as f:
        pickle.dump(result, f)

    # ===== FORMAT SUMMARY FOR AGENT =====
    summary = f"""
SCREENING RESULTS: {chemsys}
{'='*50}

Total materials queried: {total_queried}
Passed all filters: {len(ranked)}

EXCLUSION BREAKDOWN:
"""
    for reason, count in sorted(exclusion_reasons.items(), key=lambda x: -x[1]):
        pct = 100 * count / total_queried
        summary += f"  • {reason}: {count} ({pct:.1f}%)\n"

    summary += f"\nTOP 10 CANDIDATES:\n{'─'*50}\n"

    for i, c in enumerate(ranked[:10], 1):
        e_hull_mev = c.energy_above_hull * 1000 if c.energy_above_hull else 0
        gap_str = f"{c.band_gap:.2f} eV" if c.band_gap else "N/A"
        source = "ICSD" if not c.is_theoretical else "predicted"
        summary += f"""
{i:2d}. {c.material_id} ({c.formula})
    Stability: {e_hull_mev:.1f} meV above hull | Band gap: {gap_str} (DFT)
    Structure: {c.space_group} ({c.crystal_system}) | Source: {source}
"""

    summary += f"""
{'─'*50}
Full results cached at: {cache_path}
Use 'analyze_candidate(mp_id)' for detailed analysis of specific materials.
"""

    return summary


# =============================================================================
# TOOL 2: CANDIDATE ANALYSIS
# =============================================================================

@tool(parse_docstring=True)
def analyze_candidate(material_id: str) -> str:
    """
    Perform deep analysis of a specific material candidate.

    This tool retrieves the crystal structure and performs:
    - Symmetry analysis (space group, point group)
    - Coordination environment analysis (CrystalNN)
    - Band structure and DOS plotting (if available)
    - Lattice parameter extraction

    Args:
        material_id: Materials Project ID (e.g., "mp-19017")

    Returns:
        Comprehensive analysis summary with paths to generated visualizations
    """
    output_subdir = OUTPUT_DIR / material_id.replace("-", "_")
    output_subdir.mkdir(exist_ok=True)

    visualizations = []

    with MPRester() as mpr:
        # ===== STRUCTURE ANALYSIS =====
        try:
            structure = mpr.get_structure_by_material_id(material_id)
        except Exception as e:
            return f"Error retrieving structure for {material_id}: {e}"

        # Symmetry
        sga = SpacegroupAnalyzer(structure)
        space_group = sga.get_space_group_symbol()
        point_group = sga.get_point_group_symbol()
        crystal_system = sga.get_crystal_system()

        # Lattice
        lattice = structure.lattice

        # ===== COORDINATION ANALYSIS =====
        cnn = CrystalNN()
        coord_summary = {}

        for i, site in enumerate(structure):
            element = str(site.specie)
            try:
                nn_info = cnn.get_nn_info(structure, i)
                cn = len(nn_info)
                neighbors = [str(nn['site'].specie) for nn in nn_info]
                neighbor_summary = Counter(neighbors)

                if element not in coord_summary:
                    coord_summary[element] = []
                coord_summary[element].append({
                    'cn': cn,
                    'neighbors': dict(neighbor_summary)
                })
            except:
                pass

        # Average coordination
        avg_coord = {}
        for el, envs in coord_summary.items():
            avg_coord[el] = sum(e['cn'] for e in envs) / len(envs)

        # ===== ELECTRONIC STRUCTURE =====
        band_gap = None
        is_direct = None
        bs_path = None
        dos_path = None

        try:
            # Get summary for band gap info
            doc = mpr.materials.summary.get_data_by_id(material_id)
            band_gap = doc.band_gap
            is_direct = doc.is_gap_direct
        except:
            pass

        # Try to get and plot band structure
        try:
            bs = mpr.get_bandstructure_by_material_id(material_id)
            if bs:
                plotter = BSPlotter(bs)
                plotter.get_plot()
                bs_path = output_subdir / "bandstructure.png"
                plt.title(f"Band Structure: {material_id}")
                plt.savefig(bs_path, dpi=150, bbox_inches='tight')
                plt.close()
                visualizations.append(str(bs_path))
        except:
            pass

        # Try to get and plot DOS
        try:
            dos = mpr.get_dos_by_material_id(material_id)
            if dos:
                plotter = DosPlotter()
                plotter.add_dos("Total", dos)
                plotter.get_plot()
                dos_path = output_subdir / "dos.png"
                plt.title(f"Density of States: {material_id}")
                plt.savefig(dos_path, dpi=150, bbox_inches='tight')
                plt.close()
                visualizations.append(str(dos_path))
        except:
            pass

    # ===== FORMAT SUMMARY =====
    summary = f"""
DETAILED ANALYSIS: {material_id}
{'='*50}

COMPOSITION: {structure.composition.reduced_formula}
Formula units: {structure.composition.get_reduced_formula_and_factor()[1]}
Number of sites: {len(structure)}

SYMMETRY:
  Space group: {space_group} (#{sga.get_space_group_number()})
  Point group: {point_group}
  Crystal system: {crystal_system}

LATTICE PARAMETERS:
  a = {lattice.a:.4f} Å
  b = {lattice.b:.4f} Å
  c = {lattice.c:.4f} Å
  α = {lattice.alpha:.2f}°
  β = {lattice.beta:.2f}°
  γ = {lattice.gamma:.2f}°
  Volume = {lattice.volume:.2f} Å³
  Density = {structure.density:.3f} g/cm³

COORDINATION ENVIRONMENTS:
"""

    for el, avg_cn in avg_coord.items():
        # Get most common neighbor
        if el in coord_summary and coord_summary[el]:
            all_neighbors = Counter()
            for env in coord_summary[el]:
                all_neighbors.update(env['neighbors'])
            common = all_neighbors.most_common(2)
            neighbor_str = ", ".join([f"{n[0]}" for n in common])
            summary += f"  {el}: CN = {avg_cn:.1f} (coordinated by {neighbor_str})\n"

    summary += f"""
ELECTRONIC STRUCTURE:
  Band gap: {f"{band_gap:.2f} eV" if band_gap else "N/A"} (DFT-GGA, likely underestimated)
  Gap type: {"Direct" if is_direct else "Indirect" if is_direct is not None else "N/A"}
"""

    if visualizations:
        summary += f"\nGENERATED VISUALIZATIONS:\n"
        for v in visualizations:
            summary += f"  • {v}\n"
    else:
        summary += f"\nNOTE: No band structure/DOS data available in MP for this material.\n"
        summary += f"      Consider running DFT calculations for electronic structure.\n"

    return summary


# =============================================================================
# TOOL 3: STRUCTURE COMPARISON
# =============================================================================

@tool(parse_docstring=True)
def compare_structures(material_id_1: str, material_id_2: str) -> str:
    """
    Compare two crystal structures for similarity.

    Uses pymatgen's StructureMatcher to determine if two materials
    are isostructural (same structure type, different composition)
    or polymorphs (same composition, different structure).

    Args:
        material_id_1: First Materials Project ID
        material_id_2: Second Materials Project ID

    Returns:
        Comparison summary including structural similarity assessment
    """
    with MPRester() as mpr:
        try:
            s1 = mpr.get_structure_by_material_id(material_id_1)
            s2 = mpr.get_structure_by_material_id(material_id_2)
        except Exception as e:
            return f"Error retrieving structures: {e}"

    # Analyze both
    sga1 = SpacegroupAnalyzer(s1)
    sga2 = SpacegroupAnalyzer(s2)

    # Structure matching
    matcher = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5)
    is_match = matcher.fit(s1, s2)

    # Anonymous matching (ignores species, just compares geometry)
    matcher_anon = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5, 
                                     primitive_cell=False, comparator=None)

    # Check if same composition
    same_comp = (s1.composition.reduced_formula == s2.composition.reduced_formula)

    # Determine relationship
    if is_match:
        if same_comp:
            relationship = "IDENTICAL or near-identical structures"
        else:
            relationship = "ISOSTRUCTURAL (same structure type, different elements)"
    else:
        if same_comp:
            relationship = "POLYMORPHS (same composition, different structures)"
        else:
            relationship = "DIFFERENT structures"

    summary = f"""
STRUCTURE COMPARISON
{'='*50}

Material 1: {material_id_1}
  Formula: {s1.composition.reduced_formula}
  Space group: {sga1.get_space_group_symbol()} (#{sga1.get_space_group_number()})
  Crystal system: {sga1.get_crystal_system()}
  Volume: {s1.lattice.volume:.2f} Å³

Material 2: {material_id_2}
  Formula: {s2.composition.reduced_formula}
  Space group: {sga2.get_space_group_symbol()} (#{sga2.get_space_group_number()})
  Crystal system: {sga2.get_crystal_system()}
  Volume: {s2.lattice.volume:.2f} Å³

COMPARISON RESULT:
  Structure match: {"YES" if is_match else "NO"}
  Same composition: {"YES" if same_comp else "NO"}
  Relationship: {relationship}
"""

    if is_match:
        # Get the mapping
        try:
            rms = matcher.get_rms_dist(s1, s2)
            if rms:
                summary += f"  RMS distance: {rms[0]:.4f} Å\n"
        except:
            pass

    return summary


# =============================================================================
# TOOL 4: PHASE DIAGRAM
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

    with MPRester() as mpr:
        entries = mpr.get_entries_in_chemsys(elements)

    if not entries:
        return f"No entries found for chemical system {chemsys}"

    pd = PhaseDiagram(entries)

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
    plotter = PDPlotter(pd, show_unstable=show_unstable)
    fig = plotter.get_plot()
    plt.title(f"Phase Diagram: {chemsys}")

    fig_path = OUTPUT_DIR / f"phase_diagram_{chemsys.replace('-', '_')}.png"
    plt.savefig(fig_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()

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


# =============================================================================
# TOOL 5: PROPERTY CORRELATION
# =============================================================================

@tool(parse_docstring=True)
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
    cache_path = CACHE_DIR / f"{chemsys.replace('-', '_')}_screening.pkl"

    if not cache_path.exists():
        return f"No cached screening results for {chemsys}. Run batch_screen_materials first."

    with open(cache_path, 'rb') as f:
        result = pickle.load(f)

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
    import numpy as np
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


# =============================================================================
# TOOL 6: IDENTIFY DATA GAPS
# =============================================================================

@tool(parse_docstring=True)
def identify_data_gaps(chemsys: str, n_top: int = 10) -> str:
    """
    Identify missing data for top candidates that would require DFT calculations.

    Analyzes the top candidates from a screening run and identifies which
    materials are missing band structure, DOS, or other electronic structure data.

    Args:
        chemsys: Chemical system (must have been screened previously)
        n_top: Number of top candidates to check

    Returns:
        Summary of data gaps and recommendations for follow-up calculations
    """
    cache_path = CACHE_DIR / f"{chemsys.replace('-', '_')}_screening.pkl"

    if not cache_path.exists():
        return f"No cached screening results for {chemsys}. Run batch_screen_materials first."

    with open(cache_path, 'rb') as f:
        result = pickle.load(f)

    top_candidates = result.candidates[:n_top]

    gaps = {
        'missing_bandstructure': [],
        'missing_dos': [],
        'predicted_structure': [],
        'no_experimental_validation': []
    }

    with MPRester() as mpr:
        for c in top_candidates:
            # Check for electronic structure data
            try:
                es_doc = mpr.materials.electronic_structure.get_data_by_id(c.material_id)
                if not es_doc or not es_doc.bandstructure:
                    gaps['missing_bandstructure'].append(c.material_id)
                if not es_doc or not es_doc.dos:
                    gaps['missing_dos'].append(c.material_id)
            except:
                gaps['missing_bandstructure'].append(c.material_id)
                gaps['missing_dos'].append(c.material_id)

            # Check provenance
            if c.is_theoretical:
                gaps['predicted_structure'].append(c.material_id)

    summary = f"""
DATA GAPS ANALYSIS: {chemsys} (top {n_top} candidates)
{'='*50}

MISSING BAND STRUCTURE ({len(gaps['missing_bandstructure'])}/{n_top}):
"""
    if gaps['missing_bandstructure']:
        for mid in gaps['missing_bandstructure']:
            summary += f"  • {mid}\n"
    else:
        summary += "  All candidates have band structure data.\n"

    summary += f"""
MISSING DOS ({len(gaps['missing_dos'])}/{n_top}):
"""
    if gaps['missing_dos']:
        for mid in gaps['missing_dos']:
            summary += f"  • {mid}\n"
    else:
        summary += "  All candidates have DOS data.\n"

    summary += f"""
PREDICTED STRUCTURES (no ICSD entry) ({len(gaps['predicted_structure'])}/{n_top}):
"""
    if gaps['predicted_structure']:
        for mid in gaps['predicted_structure']:
            summary += f"  • {mid}\n"
    else:
        summary += "  All candidates have experimental structures.\n"

    # Recommendations
    summary += f"""
{'='*50}
RECOMMENDATIONS:
"""

    if gaps['missing_bandstructure']:
        summary += f"""
1. RUN DFT CALCULATIONS for band structures:
   Materials: {', '.join(gaps['missing_bandstructure'][:5])}
   Suggested workflow: atomate2 BandStructureMaker
"""

    if gaps['predicted_structure']:
        summary += f"""
2. EXPERIMENTAL VALIDATION needed for predicted structures:
   Materials: {', '.join(gaps['predicted_structure'][:5])}
   These have not been experimentally synthesized.
"""

    summary += """
3. BAND GAP CORRECTION:
   All band gaps are DFT-GGA (underestimated by ~30-50%).
   For accurate values, run HSE06 or GW calculations.
"""

    return summary


# =============================================================================
# TOOL 7: THINKING TOOL
# =============================================================================

@tool(parse_docstring=True)
def think(thought: str) -> str:
    """
    Record a reasoning step or reflection.

    Use this tool to think through:
    - What you've learned from previous tool results
    - What the next logical step should be
    - How to interpret findings in context of the research question

    Args:
        thought: Your reasoning or reflection

    Returns:
        Confirmation that thought was recorded
    """
    return f"Thought recorded: {thought}"


# =============================================================================
# EXPORT ALL TOOLS
# =============================================================================

ALL_TOOLS = [
    batch_screen_materials,
    analyze_candidate,
    compare_structures,
    generate_phase_diagram,
    correlate_properties,
    identify_data_gaps,
    think
]
