"""
Shared utilities for MP Deep Research tools.
Handles configuration, caching, API resilience, and common helper functions.
"""

import os
import json
import pickle
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional, Any
from pymatgen.core import Structure
from langchain_core.tools import tool
# MP imports
from mp_deep_research.models import ScreeningResult

# Placeholder for patching in tests; lazy import inside functions.
MPRester = None

# =============================================================================
# CONFIGURATION
# =============================================================================

CACHE_DIR = Path("./cache")
OUTPUT_DIR = Path("./outputs")

# Ensure directories exist
CACHE_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Cache version - bump this when MaterialCandidate schema changes
CACHE_VERSION = "v2"

# Fields we want from MP API - defined centrally for easy updates
MP_SUMMARY_FIELDS = [
    "material_id", "formula_pretty", "composition",
    "band_gap", "is_metal", "is_gap_direct",
    "formation_energy_per_atom", "energy_above_hull", "is_stable",
    "symmetry", "volume", "density", "nsites", "theoretical"
]

# Fallback fields if some are unavailable
MP_MINIMAL_FIELDS = [
    "material_id", "formula_pretty",
    "band_gap", "energy_above_hull", "is_stable"
]

# =============================================================================
# CACHING HELPERS
# =============================================================================

def get_cache_path(chemsys: str, suffix: str = "screening") -> Path:
    """Generate versioned cache path."""
    return CACHE_DIR / f"{chemsys.replace('-', '_')}_{suffix}_{CACHE_VERSION}.pkl"


def get_json_cache_path(chemsys: str, suffix: str = "screening") -> Path:
    """Generate path for raw JSON cache (schema-independent)."""
    return CACHE_DIR / f"{chemsys.replace('-', '_')}_{suffix}_raw.json"


def save_to_cache(chemsys: str, result: ScreeningResult, raw_data: List[dict]) -> Path:
    """
    Save both pickle and raw JSON cache.
    
    Args:
        chemsys: The chemical system identifier
        result: The high-level ScreeningResult object (pickled)
        raw_data: List of dictionaries containing raw API responses (JSON)
        
    Returns:
        Path object to the pickle file
    """
    # Pickle for fast loading
    pkl_path = get_cache_path(chemsys)
    with open(pkl_path, 'wb') as f:
        pickle.dump(result, f)
    
    # JSON for resilience (can rebuild if schema changes)
    json_path = get_json_cache_path(chemsys)
    with open(json_path, 'w') as f:
        json.dump({
            "chemsys": chemsys,
            "timestamp": datetime.now().isoformat(),
            "cache_version": CACHE_VERSION,
            "raw_data": raw_data
        }, f, indent=2, default=str)
    
    return pkl_path


def load_from_cache(chemsys: str) -> Optional[ScreeningResult]:
    """
    Load from cache, falling back to JSON rebuild if pickle fails.
    
    Returns:
        ScreeningResult object or None if not found/recoverable
    """
    pkl_path = get_cache_path(chemsys)
    
    if pkl_path.exists():
        try:
            with open(pkl_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            # Pickle failed (likely schema change) - try JSON rebuild
            print(f"Pickle load failed ({e}), attempting JSON rebuild...")
    
    # Try to rebuild from JSON (Placeholder for future implementation)
    json_path = get_json_cache_path(chemsys)
    if json_path.exists():
        print(f"JSON cache found for {chemsys}, but object rebuild is not fully implemented. Re-querying.")
    
    return None

# =============================================================================
# API UTILITIES
# =============================================================================

def safe_get_attr(doc: Any, attr: str, default: Any = None) -> Any:
    """
    Safely get attribute from MP doc, handling nested attributes.
    
    Args:
        doc: The document/object to access
        attr: Attribute path (e.g., "symmetry.symbol")
        default: Return value if attribute is missing
    """
    try:
        if "." in attr:
            parts = attr.split(".")
            val = doc
            for part in parts:
                val = getattr(val, part, None)
                if val is None:
                    return default
            return val
        return getattr(doc, attr, default)
    except:
        return default


def query_mp_with_resilience(chemsys: str, fields: List[str] = None) -> Tuple[List[Any], List[dict]]:
    """
    Query MP API with fallback for schema changes.
    
    Returns:
        Tuple of (docs, raw_data_dicts)
    """
    fields = fields or MP_SUMMARY_FIELDS
    
    try:
        # Import lazily to avoid import-time failures in environments
        # that don't have mp_api fully configured.
        mprester_cls = MPRester
        if mprester_cls is None:
            from mp_api.client import MPRester as mprester_cls
        with mprester_cls() as mpr:
            docs = mpr.materials.summary.search(
                chemsys=chemsys,
                fields=fields
            )
            # Convert to dicts for raw caching
            raw_data = []
            for doc in docs:
                raw_data.append({f: getattr(doc, f, None) for f in fields})
            return docs, raw_data
            
    except Exception as e:
        error_str = str(e).lower()
        
        # Handle field-related errors by trying minimal fields
        if "field" in error_str or "attribute" in error_str or "schema" in error_str:
            print(f"MP API field error: {e}")
            print("Attempting query with minimal fields...")
            
            try:
                with MPRester() as mpr:
                    docs = mpr.materials.summary.search(
                        chemsys=chemsys,
                        fields=MP_MINIMAL_FIELDS
                    )
                    raw_data = []
                    for doc in docs:
                        raw_data.append({f: getattr(doc, f, None) for f in MP_MINIMAL_FIELDS})
                    return docs, raw_data
            except Exception as e2:
                raise RuntimeError(f"MP API query failed even with minimal fields: {e2}")
        
        # Re-raise other errors (auth, network, etc.)
        raise



def ensure_structure(structure_data) -> Structure:
    """Convert structure data to Structure object if needed."""
    if structure_data is None:
        raise ValueError("Structure data is None")
    if isinstance(structure_data, dict):
        return Structure.from_dict(structure_data)
    elif isinstance(structure_data, Structure):
        return structure_data
    else:
        raise TypeError(f"Cannot convert {type(structure_data)} to Structure")


def get_structure_from_mp(material_id: str) -> Structure:
    """Safely get structure from Materials Project."""
    from mp_api.client import MPRester
    
    with MPRester() as mpr:
        structure_data = mpr.get_structure_by_material_id(material_id)
        return ensure_structure(structure_data)