# =============================================================================
# FLOW ENGINEERING IMPLEMENTATION
# =============================================================================
# Replaces fragile prompt rules with deterministic Python logic
# 
# Add to: src/mp_deep_research/tools/flows.py
# =============================================================================

from typing import Optional
from dataclasses import dataclass, field
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from pathlib import Path

# =============================================================================
# SECTION 1: STRUCTURED OUTPUT SCHEMAS (Pydantic)
# =============================================================================
# These enforce strict output formats - LLM cannot hallucinate structure

class SubstitutionResult(BaseModel):
    """Schema for substitution output - forces consistent format."""
    success: bool
    parent_formula: str
    new_formula: str
    substitutions_applied: dict[str, str]
    structure_path: str = Field(..., description="Path to saved structure file")
    volume_original: float = Field(..., description="Original volume in Å³")


class RelaxationResult(BaseModel):
    """Schema for relaxation output - includes pre-calculated meV."""
    success: bool
    formula: str
    structure_path: str = Field(..., description="Path to relaxed structure")
    
    # Energy in BOTH units - LLM doesn't need to convert
    energy_per_atom_ev: float
    energy_per_atom_mev: int = Field(..., description="Energy in meV (pre-calculated)")
    
    # Volume change with automatic flag
    volume_original: float
    volume_final: float
    volume_change_percent: float
    volume_flag: str = Field(..., description="NORMAL | HIGH_STRAIN | EXCESSIVE")
    
    converged: bool


class StabilityResult(BaseModel):
    """Schema for stability assessment - forces decomposition as list."""
    formula: str
    
    # Energy above hull in BOTH units
    e_above_hull_ev: float
    e_above_hull_mev: int = Field(..., description="Pre-calculated meV/atom")
    
    # Stability verdict (calculated by code, not LLM)
    stability_verdict: str = Field(..., description="STABLE | METASTABLE | MARGINAL | UNSTABLE")
    
    # Decomposition as LIST - prevents "decomposes into itself" hallucination
    decomposes: bool
    decomposition_products: list[str] = Field(
        default_factory=list, 
        description="List of product formulas, e.g., ['Na3PO4', 'Fe2O3']"
    )
    
    # Data provenance
    data_source: str = Field(default="ML Prediction (M3GNet)", description="DFT or ML")


class InnovationReport(BaseModel):
    """Complete innovation flow report - all results in one schema."""
    # Inputs
    parent_material_id: str
    substitutions: dict[str, str]
    
    # Theory prediction (from ionic radii)
    predicted_volume_change: str  # "expansion" | "contraction" | "minimal"
    ionic_radii_comparison: str   # e.g., "Li+(0.76Å) → Na+(1.02Å): +34% larger"
    
    # Actual results
    substitution: SubstitutionResult
    relaxation: RelaxationResult
    stability: StabilityResult
    
    # Summary (code-generated, not LLM-generated)
    theory_vs_actual: str
    recommendation: str


# =============================================================================
# SECTION 2: MIDDLEWARE LOGIC (Unit Conversion in Tools)
# =============================================================================
# All math happens HERE, not in the LLM

# Ionic radii lookup table (Shannon radii, 6-coordinate)
IONIC_RADII = {
    "Li": 0.76, "Na": 1.02, "K": 1.38, "Rb": 1.52, "Cs": 1.67,
    "Mg": 0.72, "Ca": 1.00, "Sr": 1.18, "Ba": 1.35,
    "Fe2+": 0.78, "Fe3+": 0.65, "Mn2+": 0.83, "Mn3+": 0.65, "Mn4+": 0.53,
    "Co2+": 0.75, "Co3+": 0.55, "Ni2+": 0.69, "Ni3+": 0.56,
    "Cu+": 0.77, "Cu2+": 0.73, "Zn": 0.74,
    "Al": 0.54, "Ti4+": 0.61, "V5+": 0.54, "Cr3+": 0.62,
}


def ev_to_mev(ev: float) -> int:
    """Convert eV to meV, return as integer."""
    return int(round(ev * 1000))


def calculate_volume_flag(change_percent: float) -> str:
    """Deterministic volume change classification."""
    abs_change = abs(change_percent)
    if abs_change < 5:
        return "NORMAL"
    elif abs_change < 10:
        return "HIGH_STRAIN"
    else:
        return "EXCESSIVE"


def calculate_stability_verdict(e_above_hull_mev: int) -> str:
    """Deterministic stability classification."""
    if e_above_hull_mev < 25:
        return "STABLE"
    elif e_above_hull_mev < 50:
        return "METASTABLE"
    elif e_above_hull_mev < 100:
        return "MARGINAL"
    else:
        return "UNSTABLE"


def predict_volume_change(old_element: str, new_element: str) -> tuple[str, str]:
    """
    Predict volume change based on ionic radii.
    Returns: (prediction, explanation)
    """
    r_old = IONIC_RADII.get(old_element, IONIC_RADII.get(f"{old_element}2+"))
    r_new = IONIC_RADII.get(new_element, IONIC_RADII.get(f"{new_element}2+"))
    
    if r_old is None or r_new is None:
        return "unknown", f"Ionic radii not found for {old_element} or {new_element}"
    
    ratio = (r_new - r_old) / r_old * 100
    explanation = f"{old_element}({r_old:.2f}Å) → {new_element}({r_new:.2f}Å): {ratio:+.0f}%"
    
    if ratio > 10:
        return "expansion", explanation
    elif ratio < -10:
        return "contraction", explanation
    else:
        return "minimal", explanation


# =============================================================================
# SECTION 3: DETERMINISTIC FLOW (Hardcoded Chain)
# =============================================================================
# The LLM doesn't decide what to do next - the code does

@tool
def run_innovation_flow(
    material_id: str,
    substitutions: dict[str, str]
) -> str:
    """
    Execute the COMPLETE innovation workflow as a single deterministic chain.
    
    This tool runs: substitute → relax → assess_stability automatically.
    The LLM does NOT need to chain these tools manually.
    
    Args:
        material_id: MP ID of parent structure (e.g., "mp-19017")
        substitutions: Element mapping (e.g., {"Li": "Na"})
    
    Returns:
        Complete innovation report with all results and analysis.
    
    Example:
        run_innovation_flow("mp-19017", {"Li": "Na"})
    """
    from mp_api.client import MPRester
    from pymatgen.core import Structure
    from pymatgen.transformations.standard_transformations import SubstitutionTransformation
    from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry
    import warnings
    
    # Import ML relaxer
    try:
        import matgl
        from matgl.ext.ase import Relaxer
        HAS_MATGL = True
    except ImportError:
        HAS_MATGL = False
    
    results = {}
    
    # =========================================================================
    # STEP 0: THEORY PREDICTION (Before any computation)
    # =========================================================================
    old_elem = list(substitutions.keys())[0]
    new_elem = list(substitutions.values())[0]
    predicted_change, radii_explanation = predict_volume_change(old_elem, new_elem)
    
    results["theory"] = {
        "predicted_volume_change": predicted_change,
        "ionic_radii_comparison": radii_explanation,
    }
    
    # =========================================================================
    # STEP 1: SUBSTITUTION (Deterministic - always runs)
    # =========================================================================
    try:
        with MPRester() as mpr:
            structure_data = mpr.get_structure_by_material_id(material_id)
            if isinstance(structure_data, dict):
                structure = Structure.from_dict(structure_data)
            else:
                structure = structure_data
        
        original_formula = structure.composition.reduced_formula
        original_volume = structure.volume
        
        # Apply substitution
        trans = SubstitutionTransformation(substitutions)
        new_structure = trans.apply_transformation(structure)
        new_structure.sort()
        new_formula = new_structure.composition.reduced_formula
        
        # Save structure
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        sub_filename = f"{material_id.replace('-', '_')}_sub_{'_'.join(substitutions.values())}.json"
        sub_path = output_dir / sub_filename
        new_structure.to(filename=str(sub_path))
        
        results["substitution"] = SubstitutionResult(
            success=True,
            parent_formula=original_formula,
            new_formula=new_formula,
            substitutions_applied=substitutions,
            structure_path=str(sub_path),
            volume_original=round(original_volume, 2),
        )
        
    except Exception as e:
        return f"FLOW FAILED at SUBSTITUTION step: {e}"
    
    # =========================================================================
    # STEP 2: RELAXATION (Deterministic - always runs after substitution)
    # =========================================================================
    if not HAS_MATGL:
        return "FLOW FAILED: MatGL not installed. Cannot run M3GNet relaxation."
    
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pot = matgl.load_model("M3GNet-MP-2021.2.8-PES")
            relaxer = Relaxer(potential=pot)
            relax_results = relaxer.relax(new_structure, fmax=0.05)
        
        final_structure = relax_results["final_structure"]
        final_energy = float(relax_results["trajectory"].energies[-1])
        energy_per_atom = final_energy / len(final_structure)
        
        # Calculate volume change
        final_volume = final_structure.volume
        vol_change = (final_volume - original_volume) / original_volume * 100
        
        # Save relaxed structure
        relaxed_filename = sub_filename.replace(".json", "_relaxed.json")
        relaxed_path = output_dir / relaxed_filename
        final_structure.to(filename=str(relaxed_path))
        
        results["relaxation"] = RelaxationResult(
            success=True,
            formula=new_formula,
            structure_path=str(relaxed_path),
            energy_per_atom_ev=round(energy_per_atom, 4),
            energy_per_atom_mev=ev_to_mev(energy_per_atom),
            volume_original=round(original_volume, 2),
            volume_final=round(final_volume, 2),
            volume_change_percent=round(vol_change, 2),
            volume_flag=calculate_volume_flag(vol_change),
            converged=True,
        )
        
    except Exception as e:
        return f"FLOW FAILED at RELAXATION step: {e}"
    
    # =========================================================================
    # STEP 3: STABILITY ASSESSMENT (Deterministic - always runs after relaxation)
    # =========================================================================
    try:
        with MPRester() as mpr:
            chemsys = "-".join(sorted(set(str(el) for el in final_structure.composition.elements)))
            entries = mpr.get_entries_in_chemsys(chemsys)
        
        if not entries:
            return f"FLOW FAILED: No entries found in MP for {chemsys}"
        
        pd = PhaseDiagram(entries)
        
        # Apply ML→DFT correction
        corrected_energy = energy_per_atom + 0.05
        total_energy = corrected_energy * len(final_structure)
        
        hypo_entry = PDEntry(final_structure.composition, total_energy)
        e_above_hull = pd.get_e_above_hull(hypo_entry)
        e_above_hull_mev = ev_to_mev(e_above_hull)
        
        # Get decomposition products
        decomp_products = []
        if e_above_hull > 0.001:
            try:
                decomp = pd.get_decomposition(final_structure.composition)
                decomp_products = [entry.name for entry in decomp.keys()]
            except:
                decomp_products = ["Unable to determine"]
        
        results["stability"] = StabilityResult(
            formula=new_formula,
            e_above_hull_ev=round(e_above_hull, 4),
            e_above_hull_mev=e_above_hull_mev,
            stability_verdict=calculate_stability_verdict(e_above_hull_mev),
            decomposes=e_above_hull_mev > 25,
            decomposition_products=decomp_products,
            data_source="ML Prediction (M3GNet, ±50 meV uncertainty)",
        )
        
    except Exception as e:
        return f"FLOW FAILED at STABILITY step: {e}"
    
    # =========================================================================
    # STEP 4: GENERATE REPORT (Code-generated summary, not LLM-generated)
    # =========================================================================
    sub = results["substitution"]
    rel = results["relaxation"]
    stab = results["stability"]
    theory = results["theory"]
    
    # Theory vs Actual comparison
    if theory["predicted_volume_change"] == "expansion" and rel.volume_change_percent > 0:
        theory_match = "✓ CONFIRMED: Predicted expansion, observed +{:.1f}%".format(rel.volume_change_percent)
    elif theory["predicted_volume_change"] == "contraction" and rel.volume_change_percent < 0:
        theory_match = "✓ CONFIRMED: Predicted contraction, observed {:.1f}%".format(rel.volume_change_percent)
    else:
        theory_match = "⚠ UNEXPECTED: Predicted {}, observed {:.1f}%".format(
            theory["predicted_volume_change"], rel.volume_change_percent
        )
    
    # Recommendation
    if stab.stability_verdict == "STABLE":
        recommendation = "PROMISING: Candidate for DFT validation"
    elif stab.stability_verdict == "METASTABLE":
        recommendation = "POSSIBLE: May be synthesizable under specific conditions"
    elif rel.volume_flag == "EXCESSIVE":
        recommendation = "REJECT: Excessive strain indicates structural incompatibility"
    else:
        recommendation = "UNLIKELY: Too far above hull, try different substitution"
    
    # Format final report
    report = f"""
════════════════════════════════════════════════════════════════════════════════
                        INNOVATION FLOW REPORT
════════════════════════════════════════════════════════════════════════════════

INPUTS
  Parent Material: {material_id} ({sub.parent_formula})
  Substitution: {substitutions}
  New Composition: {sub.new_formula}

────────────────────────────────────────────────────────────────────────────────
STEP 1: THEORY PREDICTION
────────────────────────────────────────────────────────────────────────────────
  Ionic Radii: {theory["ionic_radii_comparison"]}
  Predicted: {theory["predicted_volume_change"].upper()}

────────────────────────────────────────────────────────────────────────────────
STEP 2: SUBSTITUTION
────────────────────────────────────────────────────────────────────────────────
  Status: {"✓ SUCCESS" if sub.success else "✗ FAILED"}
  Structure saved: {sub.structure_path}

────────────────────────────────────────────────────────────────────────────────
STEP 3: ML RELAXATION (M3GNet)
────────────────────────────────────────────────────────────────────────────────
  Energy: {rel.energy_per_atom_mev} meV/atom ({rel.energy_per_atom_ev:.4f} eV/atom)
  Volume: {rel.volume_original:.1f} → {rel.volume_final:.1f} Ų ({rel.volume_change_percent:+.1f}%)
  Volume Flag: {rel.volume_flag}
  Relaxed structure: {rel.structure_path}

────────────────────────────────────────────────────────────────────────────────
STEP 4: STABILITY ASSESSMENT
────────────────────────────────────────────────────────────────────────────────
  E_above_hull: {stab.e_above_hull_mev} meV/atom
  Verdict: {stab.stability_verdict}
  Decomposes: {stab.decomposes}
  Products: {', '.join(stab.decomposition_products) if stab.decomposition_products else 'N/A (stable)'}
  Data Source: {stab.data_source}

════════════════════════════════════════════════════════════════════════════════
                              ANALYSIS
════════════════════════════════════════════════════════════════════════════════
  Theory vs Actual: {theory_match}
  
  RECOMMENDATION: {recommendation}
════════════════════════════════════════════════════════════════════════════════
"""
    return report


# =============================================================================
# SECTION 4: UPDATED INDIVIDUAL TOOLS (with middleware)
# =============================================================================
# If you still want individual tools, here they are with built-in unit conversion

@tool
def assess_stability_v2(structure_path: str, energy_per_atom: float) -> str:
    """
    Assess thermodynamic stability with automatic unit conversion and verdicts.
    
    All math is done in code - LLM just reads the results.
    
    Args:
        structure_path: Path to relaxed structure file
        energy_per_atom: Energy in eV/atom from relaxation
    
    Returns:
        Stability report with meV values and decomposition products.
    """
    from pymatgen.core import Structure
    from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry
    from mp_api.client import MPRester
    
    structure = Structure.from_file(structure_path)
    formula = structure.composition.reduced_formula
    
    with MPRester() as mpr:
        chemsys = "-".join(sorted(set(str(el) for el in structure.composition.elements)))
        entries = mpr.get_entries_in_chemsys(chemsys)
    
    if not entries:
        return f"ERROR: No MP entries for {chemsys}"
    
    pd = PhaseDiagram(entries)
    
    # Apply correction and calculate
    corrected = energy_per_atom + 0.05
    total = corrected * len(structure)
    entry = PDEntry(structure.composition, total)
    e_hull = pd.get_e_above_hull(entry)
    
    # ===== MIDDLEWARE: All calculations done here =====
    e_hull_mev = ev_to_mev(e_hull)
    verdict = calculate_stability_verdict(e_hull_mev)
    
    # Get decomposition products as LIST
    decomp_list = []
    if e_hull > 0.001:
        try:
            decomp = pd.get_decomposition(structure.composition)
            decomp_list = [e.name for e in decomp.keys()]
        except:
            decomp_list = ["Unknown"]
    
    return f"""
STABILITY ASSESSMENT: {formula}
══════════════════════════════════════════════
E_above_hull: {e_hull_mev} meV/atom
Verdict: {verdict}

Decomposition: {', '.join(decomp_list) if decomp_list else 'None (stable)'}

Data Source: ML Prediction (M3GNet, ±50 meV uncertainty)
"""
