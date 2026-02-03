"""
Screening tool for filtering materials from the Materials Project database.
"""

from typing import Optional, List
from collections import Counter
import inspect

from langchain_core.tools import tool

from mp_deep_research.models import MaterialCandidate, ScreeningResult
from .shared import (
    load_from_cache,
    save_to_cache,
    query_mp_with_resilience,
    safe_get_attr,
    get_cache_path,
    ensure_structure
)


_TOOL_KWARGS = {"parse_docstring": True} if "parse_docstring" in inspect.signature(tool).parameters else {}


@tool(**_TOOL_KWARGS)
def batch_screen_materials(
    chemsys: str,
    stability_threshold: float = 0.025,
    band_gap_min: Optional[float] = None,
    band_gap_max: Optional[float] = None,
    require_semiconductor: bool = False,
    use_cache: bool = True,
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
        use_cache: If True, use cached results if available (default True)
    
    Returns:
        Summary of screening results including top candidates and exclusion statistics
    """
    # ===== CHECK CACHE =====
    if use_cache:
        cached = load_from_cache(chemsys)
        if cached:
            # Note: In a production system, you might want to re-filter cached results
            # if the threshold arguments (like band_gap_min) have changed.
            # For now, we return the cached summary directly.
            return _format_screening_summary(cached, from_cache=True)
    
    # ===== QUERY PHASE (with resilience) =====
    try:
        # Uses the shared resilience wrapper to handle API schema changes
        docs, raw_data = query_mp_with_resilience(chemsys)
    except Exception as e:
        return f"Error querying Materials Project: {e}\n\nPlease check:\n1. Your MP_API_KEY is valid\n2. The chemical system '{chemsys}' is correctly formatted (e.g., 'Li-Fe-O')"
    
    if not docs:
        return f"No materials found in chemical system {chemsys}."
    
    # ===== CONVERT TO CANDIDATES =====
    all_candidates = []
    for doc in docs:
        symmetry = safe_get_attr(doc, "symmetry")
        
        candidate = MaterialCandidate(
            material_id=str(safe_get_attr(doc, "material_id", "unknown")),
            formula=safe_get_attr(doc, "formula_pretty", "?"),
            formation_energy=safe_get_attr(doc, "formation_energy_per_atom"),
            energy_above_hull=safe_get_attr(doc, "energy_above_hull"),
            is_stable=safe_get_attr(doc, "is_stable", False),
            band_gap=safe_get_attr(doc, "band_gap"),
            is_metal=safe_get_attr(doc, "is_metal", False),
            is_direct_gap=safe_get_attr(doc, "is_gap_direct"),
            space_group=safe_get_attr(symmetry, "symbol") if symmetry else None,
            space_group_number=safe_get_attr(symmetry, "number") if symmetry else None,
            crystal_system=safe_get_attr(symmetry, "crystal_system.value") if symmetry else None,
            volume=safe_get_attr(doc, "volume"),
            density=safe_get_attr(doc, "density"),
            nsites=safe_get_attr(doc, "nsites"),
            structure_source="ICSD" if not safe_get_attr(doc, "theoretical", True) else "DFT-predicted",
            is_theoretical=safe_get_attr(doc, "theoretical", True),
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
    result = ScreeningResult(
        chemsys=chemsys,
        total_queried=total_queried,
        total_passed=len(ranked),
        candidates=ranked,
        exclusion_stats=dict(exclusion_reasons),
        cache_path=str(get_cache_path(chemsys))
    )
    
    try:
        cache_path = save_to_cache(chemsys, result, raw_data)
        result.cache_path = str(cache_path)
    except Exception as e:
        print(f"Warning: Failed to cache results: {e}")
    
    return _format_screening_summary(result, from_cache=False)


def _format_screening_summary(result: ScreeningResult, from_cache: bool = False) -> str:
    """Format screening results for agent consumption."""
    cache_note = " (from cache)" if from_cache else ""
    
    summary = f"""
SCREENING RESULTS: {result.chemsys}{cache_note}
{'='*50}

Total materials queried: {result.total_queried}
Passed all filters: {result.total_passed}

EXCLUSION BREAKDOWN:
"""
    for reason, count in sorted(result.exclusion_stats.items(), key=lambda x: -x[1]):
        pct = 100 * count / result.total_queried if result.total_queried > 0 else 0
        summary += f"  • {reason}: {count} ({pct:.1f}%)\n"
    
    summary += f"\nTOP 10 CANDIDATES:\n{'─'*50}\n"
    
    for i, c in enumerate(result.candidates[:10], 1):
        e_hull_mev = c.energy_above_hull * 1000 if c.energy_above_hull else 0
        gap_str = f"{c.band_gap:.2f} eV" if c.band_gap else "N/A"
        source = "ICSD" if not c.is_theoretical else "predicted"
        summary += f"""
{i:2d}. {c.material_id} ({c.formula})
    Stability: {e_hull_mev:.1f} meV above hull | Band gap: {gap_str} (DFT)
    Structure: {c.space_group or 'N/A'} ({c.crystal_system or 'N/A'}) | Source: {source}
"""
    
    summary += f"""
{'─'*50}
Cache: {result.cache_path}
Use 'analyze_candidate(mp_id)' for detailed analysis of specific materials.
"""
    
    return summary