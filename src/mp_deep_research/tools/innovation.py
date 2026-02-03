"""
Innovation tools for designing and simulating new materials.
"""
from typing import Dict, Optional
from langchain_core.tools import tool
from pydantic import BaseModel, Field
import warnings
import os
from .shared import ensure_structure, get_structure_from_mp

# Pymatgen imports
from pymatgen.core import Structure, Composition
from pymatgen.transformations.standard_transformations import SubstitutionTransformation
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry

# MatGL imports
try:
    import matgl
    from matgl.ext.ase import Relaxer
    import torch
    HAS_MATGL = True
except ImportError:
    HAS_MATGL = False
    Relaxer = None
    matgl = None
    torch = None

from .shared import OUTPUT_DIR

# --- SCHEMAS ---
class SubstituteSchema(BaseModel):
    material_id: str = Field(description="The MP ID of the parent structure (e.g., 'mp-19017')")
    substitutions: Dict[str, str] = Field(description="Dictionary of substitutions (e.g., {'Fe': 'Mn'})")
    structure_json: Optional[str] = Field(default=None, description="Optional JSON string of a structure")

class RelaxSchema(BaseModel):
    structure_path: str = Field(description="Path to the structure file (JSON/CIF) to relax")
    fmax: float = Field(default=0.05, description="Force convergence threshold in eV/A")
    estimate_uncertainty: bool = Field(default=True, description="Run ensemble relaxation to estimate error bars?")

class StabilitySchema(BaseModel):
    composition: str = Field(description="The reduced formula (e.g., 'LiFePO4')")
    energy_per_atom: float = Field(description="Total energy per atom in eV")

# =============================================================================
# TOOL 1: SUBSTITUTE SPECIES
# =============================================================================

@tool(args_schema=SubstituteSchema)
def substitute_species(
    material_id: str,
    substitutions: Dict[str, str],
    structure_json: Optional[str] = None
) -> str:
    """
    Create a new material by chemically substituting elements in an existing structure.
    Args:
        material_id: MP ID of parent structure (e.g., "mp-19017")
        substitutions: Dict mapping old element to new (e.g., {"Li": "Na", "Fe": "Mn"})
        structure_json: Optional JSON string if not fetching from MP
    
    Returns:
        Summary of substitution and path to new structure file.
    
    Example:
        substitute_species("mp-19017", {"Li": "Na"})  # LiFePO4 → NaFePO4
    """
    if not substitutions:
        return """ERROR: 'substitutions' argument is REQUIRED.
    
    from mp_api.client import MPRester

    # 1. Get Parent Structure
    try:
        if structure_json:
            structure = Structure.from_json(structure_json)
        else:
            structure = get_structure_from_mp(material_id)

        # --- FIX: Ensure it is a Pymatgen Object, not a Dict ---
        if isinstance(structure, dict):
            structure = Structure.from_dict(structure)
        # -----------------------------------------------------------
                
    except Exception as e:
        return f"Error retrieving parent structure: {e}"
    
    # 2. Perform Substitution
    try:
        # Check if structure is valid
        if not hasattr(structure, 'replace_species'):
            return f"Error: Retrieved data is type {type(structure)}, expected Structure."

        trans = SubstitutionTransformation(substitutions)
        new_structure = trans.apply_transformation(structure)
        new_structure.sort()
        
    except Exception as e:
        return f"Error performing substitution: {e}"
    
    # 3. Save to output
    filename = f"{material_id}_sub_{'_'.join(substitutions.values())}.json"
    out_path = OUTPUT_DIR / filename
    new_structure.to(filename=str(out_path))
    
    return f"""
SUBSTITUTION SUCCESSFUL:
{'='*50}
Parent Material: {material_id} ({structure.composition.reduced_formula})
New Composition: {new_structure.composition.reduced_formula}
Substitutions: {substitutions}

Structure saved to: {out_path}

NEXT STEP:
Run 'relax_structure_m3gnet' on this structure to check if it is stable.
"""

# =============================================================================
# TOOL 2: M3GNET RELAXATION (Self-Healing)
# =============================================================================

@tool(args_schema=RelaxSchema)
def relax_structure_m3gnet(
    structure_path: str,
    fmax: float = 0.05,
    estimate_uncertainty: bool = True
) -> str:
    """
    Simulate structure using M3GNet with optional Uncertainty Quantification (UQ).
    """
    if not HAS_MATGL:
        return "Error: 'matgl' library not installed."
    
    try:
        import numpy as np
        # 1. Load Structure
        original_structure = Structure.from_file(structure_path)
        original_vol = original_structure.volume
        
        # 2. Load Model (With Self-Healing Cache)
        model_name = "M3GNet-MP-2021.2.8-PES"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                pot = matgl.load_model(model_name)
            except Exception as e:
                # --- FIX: Auto-clear cache if model load fails ---
                print(f"Warning: Model load failed ({e}). Clearing cache and retrying...")
                try:
                    matgl.clear_cache() 
                except:
                    pass # Keep going if clear_cache fails (might not exist)
                pot = matgl.load_model(model_name)
                # -------------------------------------------------

            relaxer = Relaxer(potential=pot)
            
            # 3. Run Ensemble Relaxation
            runs = []
            
            # Run 1: Standard
            res_std = relaxer.relax(original_structure, fmax=fmax)
            runs.append(float(res_std["trajectory"].energies[-1]))
            final_structure = res_std["final_structure"]
            
            if estimate_uncertainty:
                # Run 2 & 3: Perturbed (Rattled)
                for _ in range(2):
                    struct_copy = original_structure.copy()
                    struct_copy.perturb(distance=0.02)
                    res_pert = relaxer.relax(struct_copy, fmax=fmax)
                    runs.append(float(res_pert["trajectory"].energies[-1]))

        # 4. Calculate Stats
        avg_energy = np.mean(runs)
        std_energy = np.std(runs) if estimate_uncertainty else 0.0
        
        # 5. Sanity Checks
        final_vol = final_structure.volume
        vol_change = (final_vol - original_vol) / original_vol * 100
        
        confidence_label = "HIGH"
        warnings_list = []
        
        if abs(vol_change) > 20:
            confidence_label = "LOW"
            warnings_list.append(f"CRITICAL: Massive volume change ({vol_change:+.1f}%). Structure likely collapsed.")
        
        if std_energy > 0.05: 
            confidence_label = "MEDIUM/LOW"
            warnings_list.append(f"CAUTION: High energy variance ({std_energy:.3f} eV). Result depends on starting position.")

        # 6. Save & Report
        save_path = structure_path.replace(".json", "_relaxed.json")
        final_structure.to(filename=save_path)
        
        energy_per_atom = avg_energy / len(final_structure)
        
        return f"""
SIMULATION RESULTS (M3GNet + UQ):
{'='*50}
Input: {structure_path}
Final Composition: {final_structure.composition.reduced_formula}

ENERGETICS:
  Total Energy: {avg_energy:.4f} +/- {std_energy:.4f} eV
  Energy per Atom: {energy_per_atom:.4f} eV/atom
  Confidence: {confidence_label}

GEOMETRY:
  Volume Change: {vol_change:+.2f}% (Original: {original_vol:.1f} -> Final: {final_vol:.1f} A^3)

WARNINGS:
  {chr(10).join(warnings_list) if warnings_list else "None - Simulation looks physically valid."}

Saved relaxed structure to: {save_path}
"""

    except Exception as e:
        return f"Simulation failed: {e}"

# =============================================================================
# TOOL 3: STABILITY ASSESSMENT
# =============================================================================

@tool(args_schema=StabilitySchema)
def assess_stability(
    composition: str,
    energy_per_atom: float
) -> str:
    """
    Check if a hypothetical material is stable by comparing it to the convex hull.
    """
    from mp_api.client import MPRester

    try:
        comp = Composition(composition)
        elements = [str(e) for e in comp.elements]
        chemsys = "-".join(sorted(elements))
        
        # 1. Fetch Competitive Phases
        with MPRester() as mpr:
            entries = mpr.get_entries_in_chemsys(elements)
            
        if not entries:
            return f"Error: No reference data found for system {chemsys}."

        # 2. Build Phase Diagram
        candidate_entry = PDEntry(comp, energy_per_atom * comp.num_atoms)
        all_entries = entries + [candidate_entry]
        pd = PhaseDiagram(all_entries)
        
        # 3. Calculate Stability
        e_above_hull = pd.get_e_above_hull(candidate_entry)
        is_stable = e_above_hull <= 1e-6
        
        decomp_str = ""
        if not is_stable:
            decomp = pd.get_decomposition(comp)
            decomp_parts = [f"{k.name} ({v*100:.1f}%)" for k, v in decomp.items()]
            decomp_str = "Decomposes into: " + ", ".join(decomp_parts)

        return f"""
STABILITY ASSESSMENT:
{'='*50}
Candidate: {composition}
Energy: {energy_per_atom:.4f} eV/atom

RESULTS:
  Energy Above Hull: {e_above_hull:.4f} eV/atom
  Is Stable? {"YES (Stable)" if is_stable else "NO (Metastable/Unstable)"}
  {decomp_str}
"""

    except Exception as e:
        return f"Stability assessment failed: {e}"