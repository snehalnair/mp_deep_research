# src/mp_deep_research/tools/analysis.py

from langchain_core.tools import tool
from pymatgen.core import Structure
# Ensure OUTPUT_DIR is imported to save files
from .shared import ensure_structure, OUTPUT_DIR 


@tool
def analyze_candidate(material_id: str) -> str:
    """
    Analyze a material candidate in detail and save its structure for simulation.
    
    Args:
        material_id: The MP ID (e.g., "mp-18767")
    
    Returns:
        Detailed analysis of structure, symmetry, and properties.
    """
    from mp_api.client import MPRester
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    
    try:
        with MPRester() as mpr:
            # Get summary document
            docs = mpr.materials.summary.search(material_ids=[material_id])
            
            if not docs:
                return f"Material {material_id} not found in Materials Project."
            
            doc = docs[0]
            
            # FIX 1: Convert structure to proper object
            structure = ensure_structure(doc.structure)
            
            # FIX 2: SAVE STRUCTURE AUTOMATICALLY
            # This prevents "No such file" errors when the agent tries to run simulations later.
            filename = f"{material_id}.json"
            file_path = OUTPUT_DIR / filename
            structure.to(filename=str(file_path))
            
            # Get symmetry info
            sga = SpacegroupAnalyzer(structure)
            spacegroup = sga.get_space_group_symbol()
            crystal_system = sga.get_crystal_system()
            
            # Get coordination environments (with error handling)
            coord_info = []
            try:
                from pymatgen.analysis.local_env import CrystalNN
                cnn = CrystalNN()
                
                seen_species = set()
                for i, site in enumerate(structure):
                    species = str(site.specie)
                    if species not in seen_species:
                        seen_species.add(species)
                        try:
                            cn_dict = cnn.get_cn_dict(structure, i)
                            cn = sum(cn_dict.values())
                            neighbors = list(cn_dict.keys())
                            coord_info.append(f"  {species}: CN={cn:.0f}, bonded to {neighbors}")
                        except Exception:
                            coord_info.append(f"  {species}: coordination analysis unavailable")
            except Exception as e:
                coord_info.append(f"  Coordination analysis failed: {e}")
            
            # Build result
            formula = structure.composition.reduced_formula
            
            result = f"""
MATERIAL ANALYSIS: {material_id}
{'='*50}
Structure saved to: {file_path}

COMPOSITION:
  Formula: {formula}
  Elements: {', '.join([str(e) for e in structure.composition.elements])}
  Num Sites: {len(structure)}

CRYSTAL STRUCTURE:
  Space Group: {spacegroup}
  Crystal System: {crystal_system}
  Lattice Parameters:
    a = {structure.lattice.a:.3f} Å
    b = {structure.lattice.b:.3f} Å
    c = {structure.lattice.c:.3f} Å
    α = {structure.lattice.alpha:.1f}°
    β = {structure.lattice.beta:.1f}°
    γ = {structure.lattice.gamma:.1f}°
  Volume: {structure.volume:.2f} Å³
  Density: {structure.density:.3f} g/cm³

COORDINATION ENVIRONMENTS:
{chr(10).join(coord_info) if coord_info else '  Analysis unavailable'}

PROPERTIES (from Materials Project - DFT validated):
  Energy Above Hull: {getattr(doc, 'energy_above_hull', 'N/A')} eV/atom
  Formation Energy: {getattr(doc, 'formation_energy_per_atom', 'N/A')} eV/atom
  Band Gap: {getattr(doc, 'band_gap', 'N/A')} eV
  Is Stable: {getattr(doc, 'is_stable', 'N/A')}
  Is Metal: {getattr(doc, 'is_metal', 'N/A')}
"""
            return result
            
    except Exception as e:
        return f"Error analyzing {material_id}: {e}"