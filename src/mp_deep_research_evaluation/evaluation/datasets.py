"""
Dataset Generation and Management for Evaluation

This module provides:
1. Ground truth dataset creation from Materials Project
2. DFT validation datasets
3. Synthetic test case generation
4. Dataset loading and preprocessing utilities
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import json
import pickle
from datetime import datetime
import random


@dataclass
class MaterialRecord:
    """A single material record for evaluation datasets."""
    material_id: str
    formula: str
    e_above_hull: float  # meV/atom
    formation_energy: float  # eV/atom
    band_gap: Optional[float]  # eV
    is_stable: bool
    is_metal: bool
    space_group: str
    crystal_system: str
    density: float
    volume: float
    nsites: int
    is_theoretical: bool
    source: str  # "ICSD", "DFT-predicted"


@dataclass
class SubstitutionRecord:
    """Ground truth for a substitution operation."""
    parent_id: str
    parent_formula: str
    substitutions: Dict[str, str]
    child_formula: str
    
    # DFT results (if available)
    dft_energy_per_atom: Optional[float] = None
    dft_e_above_hull: Optional[float] = None
    dft_volume_change_percent: Optional[float] = None
    dft_is_stable: Optional[bool] = None
    
    # ML prediction (for comparison)
    ml_energy_per_atom: Optional[float] = None
    ml_e_above_hull: Optional[float] = None
    
    # Metadata
    reference_doi: Optional[str] = None
    experimentally_verified: bool = False


@dataclass
class EvaluationDataset:
    """Complete evaluation dataset."""
    name: str
    version: str
    created_at: str
    description: str
    
    # Data
    materials: List[MaterialRecord] = field(default_factory=list)
    substitutions: List[SubstitutionRecord] = field(default_factory=list)
    
    # Metadata
    chemical_systems: List[str] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "version": self.version,
            "created_at": self.created_at,
            "description": self.description,
            "n_materials": len(self.materials),
            "n_substitutions": len(self.substitutions),
            "chemical_systems": self.chemical_systems,
            "statistics": self.statistics,
        }


# =============================================================================
# DISCOVERY BENCHMARK DATASET
# =============================================================================

def create_discovery_benchmark_dataset(
    chemical_systems: Optional[List[str]] = None,
    use_api: bool = True,
    cache_path: Optional[str] = None,
) -> EvaluationDataset:
    """
    Create a discovery benchmark dataset from Materials Project.
    
    Args:
        chemical_systems: List of chemical systems to include
        use_api: Whether to fetch from MP API (requires API key)
        cache_path: Optional path to cache/load data
    
    Returns:
        EvaluationDataset with ground truth materials
    """
    # Default systems covering common applications
    if chemical_systems is None:
        chemical_systems = [
            # Battery cathodes
            "Li-Co-O",
            "Li-Fe-P-O",
            "Li-Mn-O",
            "Li-Ni-Mn-Co-O",
            "Na-Fe-P-O",
            # Thermoelectrics
            "Bi-Te",
            "Pb-Te",
            "Si-Ge",
            # Solar absorbers
            "Cu-Zn-Sn-S",
            "Cu-In-Ga-Se",
            "Cs-Pb-I",
            # Transparent conductors
            "In-Sn-O",
            "Zn-O",
            # Solid electrolytes
            "Li-La-Zr-O",
            "Li-P-S",
        ]
    
    dataset = EvaluationDataset(
        name="MP_Discovery_Benchmark",
        version="1.0.0",
        created_at=datetime.now().isoformat(),
        description="Ground truth materials from Materials Project for discovery benchmark",
        chemical_systems=chemical_systems,
    )
    
    # Try to load from cache
    if cache_path and Path(cache_path).exists():
        return _load_dataset(cache_path)
    
    # Fetch from API
    if use_api:
        try:
            from mp_api.client import MPRester
            
            with MPRester() as mpr:
                for chemsys in chemical_systems:
                    try:
                        docs = mpr.materials.summary.search(
                            chemsys=chemsys,
                            fields=[
                                "material_id", "formula_pretty",
                                "energy_above_hull", "formation_energy_per_atom",
                                "band_gap", "is_stable", "is_metal",
                                "symmetry", "density", "volume", "nsites",
                                "theoretical",
                            ]
                        )
                        
                        for doc in docs:
                            symmetry = getattr(doc, "symmetry", None)
                            record = MaterialRecord(
                                material_id=str(doc.material_id),
                                formula=doc.formula_pretty or "?",
                                e_above_hull=(doc.energy_above_hull or 0) * 1000,  # Convert to meV
                                formation_energy=doc.formation_energy_per_atom or 0,
                                band_gap=doc.band_gap,
                                is_stable=doc.is_stable or False,
                                is_metal=doc.is_metal or False,
                                space_group=getattr(symmetry, "symbol", "?") if symmetry else "?",
                                crystal_system=str(getattr(symmetry, "crystal_system", "?")) if symmetry else "?",
                                density=doc.density or 0,
                                volume=doc.volume or 0,
                                nsites=doc.nsites or 0,
                                is_theoretical=doc.theoretical if doc.theoretical is not None else True,
                                source="ICSD" if not doc.theoretical else "DFT-predicted",
                            )
                            dataset.materials.append(record)
                            
                    except Exception as e:
                        print(f"Warning: Failed to fetch {chemsys}: {e}")
                        
        except ImportError:
            print("mp_api not installed. Using synthetic data.")
            dataset = _create_synthetic_discovery_data(chemical_systems)
    else:
        dataset = _create_synthetic_discovery_data(chemical_systems)
    
    # Compute statistics
    dataset.statistics = _compute_dataset_statistics(dataset)
    
    # Cache if path provided
    if cache_path:
        _save_dataset(dataset, cache_path)
    
    return dataset


def _create_synthetic_discovery_data(
    chemical_systems: List[str]
) -> EvaluationDataset:
    """Create synthetic data for testing without API access."""
    
    # Well-known materials for testing
    known_materials = [
        MaterialRecord(
            material_id="mp-22526",
            formula="LiCoO2",
            e_above_hull=0,
            formation_energy=-2.38,
            band_gap=2.7,
            is_stable=True,
            is_metal=False,
            space_group="R-3m",
            crystal_system="trigonal",
            density=5.05,
            volume=32.9,
            nsites=4,
            is_theoretical=False,
            source="ICSD",
        ),
        MaterialRecord(
            material_id="mp-19017",
            formula="LiFePO4",
            e_above_hull=0,
            formation_energy=-1.85,
            band_gap=3.8,
            is_stable=True,
            is_metal=False,
            space_group="Pnma",
            crystal_system="orthorhombic",
            density=3.6,
            volume=291.4,
            nsites=28,
            is_theoretical=False,
            source="ICSD",
        ),
        MaterialRecord(
            material_id="mp-18767",
            formula="LiMnPO4",
            e_above_hull=0,
            formation_energy=-1.79,
            band_gap=4.1,
            is_stable=True,
            is_metal=False,
            space_group="Pnma",
            crystal_system="orthorhombic",
            density=3.4,
            volume=302.1,
            nsites=28,
            is_theoretical=False,
            source="ICSD",
        ),
        MaterialRecord(
            material_id="mp-34202",
            formula="Bi2Te3",
            e_above_hull=0,
            formation_energy=-0.21,
            band_gap=0.15,
            is_stable=True,
            is_metal=False,
            space_group="R-3m",
            crystal_system="trigonal",
            density=7.86,
            volume=159.8,
            nsites=5,
            is_theoretical=False,
            source="ICSD",
        ),
        MaterialRecord(
            material_id="mp-20232",
            formula="In2O3",
            e_above_hull=0,
            formation_energy=-3.02,
            band_gap=2.9,
            is_stable=True,
            is_metal=False,
            space_group="Ia-3",
            crystal_system="cubic",
            density=7.18,
            volume=64.8,
            nsites=10,
            is_theoretical=False,
            source="ICSD",
        ),
    ]
    
    dataset = EvaluationDataset(
        name="Synthetic_Discovery_Data",
        version="1.0.0",
        created_at=datetime.now().isoformat(),
        description="Synthetic ground truth for testing (no API required)",
        materials=known_materials,
        chemical_systems=chemical_systems,
    )
    
    return dataset


# =============================================================================
# INNOVATION BENCHMARK DATASET
# =============================================================================

def create_innovation_benchmark_dataset(
    substitution_pairs: Optional[List[Dict]] = None,
    include_dft_validation: bool = True,
    cache_path: Optional[str] = None,
) -> EvaluationDataset:
    """
    Create an innovation benchmark dataset with substitution ground truth.
    
    Args:
        substitution_pairs: List of substitution definitions
        include_dft_validation: Include DFT validation data where available
        cache_path: Optional cache path
    
    Returns:
        EvaluationDataset with substitution ground truth
    """
    
    # Default substitution test cases
    if substitution_pairs is None:
        substitution_pairs = [
            # Olivine cathode variants
            {"parent_id": "mp-19017", "parent_formula": "LiFePO4", "subs": {"Li": "Na"}},
            {"parent_id": "mp-19017", "parent_formula": "LiFePO4", "subs": {"Fe": "Mn"}},
            {"parent_id": "mp-19017", "parent_formula": "LiFePO4", "subs": {"Fe": "Co"}},
            {"parent_id": "mp-19017", "parent_formula": "LiFePO4", "subs": {"Fe": "Ni"}},
            {"parent_id": "mp-19017", "parent_formula": "LiFePO4", "subs": {"Li": "Na", "Fe": "Mn"}},
            
            # Layered oxide variants
            {"parent_id": "mp-22526", "parent_formula": "LiCoO2", "subs": {"Co": "Mn"}},
            {"parent_id": "mp-22526", "parent_formula": "LiCoO2", "subs": {"Co": "Ni"}},
            {"parent_id": "mp-22526", "parent_formula": "LiCoO2", "subs": {"Li": "Na"}},
            
            # Thermoelectric variants
            {"parent_id": "mp-34202", "parent_formula": "Bi2Te3", "subs": {"Te": "Se"}},
            {"parent_id": "mp-34202", "parent_formula": "Bi2Te3", "subs": {"Bi": "Sb"}},
        ]
    
    dataset = EvaluationDataset(
        name="MP_Innovation_Benchmark",
        version="1.0.0",
        created_at=datetime.now().isoformat(),
        description="Substitution ground truth for innovation benchmark",
    )
    
    # DFT validation data (from literature / previous calculations)
    # These are approximate values based on published DFT studies
    dft_validation = {
        ("mp-19017", "Li", "Na"): {
            "child_formula": "NaFePO4",
            "dft_e_above_hull": 45,
            "dft_energy_per_atom": -6.832,
            "dft_volume_change_percent": 12.5,
            "dft_is_stable": False,
        },
        ("mp-19017", "Fe", "Mn"): {
            "child_formula": "LiMnPO4",
            "dft_e_above_hull": 0,
            "dft_energy_per_atom": -6.754,
            "dft_volume_change_percent": 2.3,
            "dft_is_stable": True,
        },
        ("mp-19017", "Fe", "Co"): {
            "child_formula": "LiCoPO4",
            "dft_e_above_hull": 15,
            "dft_energy_per_atom": -6.789,
            "dft_volume_change_percent": -3.1,
            "dft_is_stable": True,
        },
        ("mp-19017", "Fe", "Ni"): {
            "child_formula": "LiNiPO4",
            "dft_e_above_hull": 35,
            "dft_energy_per_atom": -6.712,
            "dft_volume_change_percent": -5.2,
            "dft_is_stable": False,
        },
        ("mp-22526", "Co", "Mn"): {
            "child_formula": "LiMnO2",
            "dft_e_above_hull": 15,
            "dft_energy_per_atom": -5.621,
            "dft_volume_change_percent": 3.8,
            "dft_is_stable": True,
        },
        ("mp-22526", "Co", "Ni"): {
            "child_formula": "LiNiO2",
            "dft_e_above_hull": 0,
            "dft_energy_per_atom": -5.734,
            "dft_volume_change_percent": 2.1,
            "dft_is_stable": True,
        },
        ("mp-22526", "Li", "Na"): {
            "child_formula": "NaCoO2",
            "dft_e_above_hull": 0,
            "dft_energy_per_atom": -5.543,
            "dft_volume_change_percent": 15.3,
            "dft_is_stable": True,
        },
    }
    
    # Create records
    for pair in substitution_pairs:
        parent_id = pair["parent_id"]
        parent_formula = pair["parent_formula"]
        subs = pair["subs"]
        
        # Make key for DFT lookup
        if len(subs) == 1:
            old_elem = list(subs.keys())[0]
            new_elem = list(subs.values())[0]
            key = (parent_id, old_elem, new_elem)
        else:
            key = None  # Multi-substitution - no DFT data
        
        # Get DFT data if available
        dft_data = dft_validation.get(key, {}) if include_dft_validation else {}
        
        # Compute child formula
        child_formula = dft_data.get("child_formula", _compute_substituted_formula(parent_formula, subs))
        
        record = SubstitutionRecord(
            parent_id=parent_id,
            parent_formula=parent_formula,
            substitutions=subs,
            child_formula=child_formula,
            dft_energy_per_atom=dft_data.get("dft_energy_per_atom"),
            dft_e_above_hull=dft_data.get("dft_e_above_hull"),
            dft_volume_change_percent=dft_data.get("dft_volume_change_percent"),
            dft_is_stable=dft_data.get("dft_is_stable"),
        )
        
        dataset.substitutions.append(record)
    
    # Compute statistics
    dataset.statistics = {
        "n_substitutions": len(dataset.substitutions),
        "n_with_dft_validation": sum(1 for s in dataset.substitutions if s.dft_e_above_hull is not None),
        "substitution_types": _count_substitution_types(dataset.substitutions),
    }
    
    if cache_path:
        _save_dataset(dataset, cache_path)
    
    return dataset


def _compute_substituted_formula(formula: str, subs: Dict[str, str]) -> str:
    """Compute new formula after substitution (simplified)."""
    result = formula
    for old, new in subs.items():
        result = result.replace(old, new)
    return result


def _count_substitution_types(records: List[SubstitutionRecord]) -> Dict[str, int]:
    """Count substitution types (e.g., alkali→alkali, TM→TM)."""
    alkali = {"Li", "Na", "K", "Rb", "Cs"}
    alkaline = {"Mg", "Ca", "Sr", "Ba"}
    transition_metals = {"Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn"}
    
    counts = {"alkali_swap": 0, "tm_swap": 0, "other": 0}
    
    for record in records:
        for old, new in record.substitutions.items():
            if old in alkali and new in alkali:
                counts["alkali_swap"] += 1
            elif old in transition_metals and new in transition_metals:
                counts["tm_swap"] += 1
            else:
                counts["other"] += 1
    
    return counts


# =============================================================================
# DATASET I/O
# =============================================================================

def _save_dataset(dataset: EvaluationDataset, path: str) -> None:
    """Save dataset to file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to serializable format
    data = {
        "metadata": dataset.to_dict(),
        "materials": [vars(m) for m in dataset.materials],
        "substitutions": [
            {
                "parent_id": s.parent_id,
                "parent_formula": s.parent_formula,
                "substitutions": s.substitutions,
                "child_formula": s.child_formula,
                "dft_energy_per_atom": s.dft_energy_per_atom,
                "dft_e_above_hull": s.dft_e_above_hull,
                "dft_volume_change_percent": s.dft_volume_change_percent,
                "dft_is_stable": s.dft_is_stable,
                "ml_energy_per_atom": s.ml_energy_per_atom,
                "ml_e_above_hull": s.ml_e_above_hull,
                "reference_doi": s.reference_doi,
                "experimentally_verified": s.experimentally_verified,
            }
            for s in dataset.substitutions
        ],
    }
    
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _load_dataset(path: str) -> EvaluationDataset:
    """Load dataset from file."""
    with open(path) as f:
        data = json.load(f)
    
    materials = [MaterialRecord(**m) for m in data.get("materials", [])]
    substitutions = [SubstitutionRecord(**s) for s in data.get("substitutions", [])]
    
    metadata = data.get("metadata", {})
    
    return EvaluationDataset(
        name=metadata.get("name", "Loaded Dataset"),
        version=metadata.get("version", "1.0.0"),
        created_at=metadata.get("created_at", datetime.now().isoformat()),
        description=metadata.get("description", ""),
        materials=materials,
        substitutions=substitutions,
        chemical_systems=metadata.get("chemical_systems", []),
        statistics=metadata.get("statistics", {}),
    )


def _compute_dataset_statistics(dataset: EvaluationDataset) -> Dict:
    """Compute statistics for a dataset."""
    materials = dataset.materials
    
    if not materials:
        return {"n_materials": 0}
    
    stats = {
        "n_materials": len(materials),
        "n_stable": sum(1 for m in materials if m.is_stable),
        "n_from_icsd": sum(1 for m in materials if m.source == "ICSD"),
        "n_theoretical": sum(1 for m in materials if m.is_theoretical),
        "n_metals": sum(1 for m in materials if m.is_metal),
        "n_semiconductors": sum(1 for m in materials if not m.is_metal and m.band_gap and m.band_gap > 0),
        "avg_e_above_hull_mev": sum(m.e_above_hull for m in materials) / len(materials),
        "crystal_systems": list(set(m.crystal_system for m in materials)),
    }
    
    return stats


def load_ground_truth_data(
    dataset_name: str = "discovery",
    cache_dir: str = "./data/evaluation"
) -> EvaluationDataset:
    """
    Convenience function to load ground truth datasets.
    
    Args:
        dataset_name: "discovery" or "innovation"
        cache_dir: Directory for cached datasets
    
    Returns:
        EvaluationDataset
    """
    cache_path = Path(cache_dir) / f"{dataset_name}_benchmark.json"
    
    if cache_path.exists():
        return _load_dataset(str(cache_path))
    
    # Create fresh dataset
    if dataset_name == "discovery":
        return create_discovery_benchmark_dataset(cache_path=str(cache_path))
    elif dataset_name == "innovation":
        return create_innovation_benchmark_dataset(cache_path=str(cache_path))
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


# =============================================================================
# DATASET SPLITTING
# =============================================================================

def split_dataset(
    dataset: EvaluationDataset,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[EvaluationDataset, EvaluationDataset, EvaluationDataset]:
    """
    Split a dataset into train/val/test sets.
    
    Useful for ML experiments within the evaluation framework.
    """
    random.seed(seed)
    
    # Split materials
    materials = list(dataset.materials)
    random.shuffle(materials)
    
    n = len(materials)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    train_materials = materials[:n_train]
    val_materials = materials[n_train:n_train + n_val]
    test_materials = materials[n_train + n_val:]
    
    # Split substitutions similarly
    substitutions = list(dataset.substitutions)
    random.shuffle(substitutions)
    
    n = len(substitutions)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    train_subs = substitutions[:n_train]
    val_subs = substitutions[n_train:n_train + n_val]
    test_subs = substitutions[n_train + n_val:]
    
    # Create split datasets
    def make_split(name_suffix, mats, subs):
        return EvaluationDataset(
            name=f"{dataset.name}_{name_suffix}",
            version=dataset.version,
            created_at=datetime.now().isoformat(),
            description=f"{name_suffix} split of {dataset.name}",
            materials=mats,
            substitutions=subs,
            chemical_systems=dataset.chemical_systems,
        )
    
    return (
        make_split("train", train_materials, train_subs),
        make_split("val", val_materials, val_subs),
        make_split("test", test_materials, test_subs),
    )
