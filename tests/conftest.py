"""
Shared test fixtures for MP Deep Research.

This file defines 'fake' data objects (fixtures) that are automatically
available to any test file in this directory.
"""

import pytest
from unittest.mock import MagicMock
from pathlib import Path
import shutil

# =============================================================================
# MOCK SCIENTIFIC OBJECTS (Pymatgen / MP API)
# =============================================================================

@pytest.fixture
def mock_structure():
    """
    Creates a fake pymatgen Structure object.
    Useful for testing analysis tools without running heavy crystal math.
    """
    struct = MagicMock()
    
    # Mock Composition
    struct.composition.reduced_formula = "LiFePO4"
    struct.composition.get_reduced_formula_and_factor.return_value = ("LiFePO4", 1)
    
    # Mock Lattice
    struct.lattice.a = 5.0
    struct.lattice.b = 5.0
    struct.lattice.c = 10.0
    struct.lattice.alpha = 90
    struct.lattice.beta = 90
    struct.lattice.gamma = 90
    struct.lattice.volume = 250.0
    
    # Mock Density & Sites
    struct.density = 3.5
    struct.__len__.return_value = 28  # len(structure)
    
    return struct


@pytest.fixture
def mock_mp_docs():
    """
    Creates a list of fake Materials Project Summary Docs.
    Useful for testing screening logic (filtering stable vs unstable).
    """
    # 1. A perfect candidate (Stable, Semiconductor)
    doc1 = MagicMock()
    doc1.material_id = "mp-1"
    doc1.formula_pretty = "LiFePO4"
    doc1.energy_above_hull = 0.0
    doc1.is_stable = True
    doc1.band_gap = 3.2
    doc1.is_metal = False
    doc1.is_gap_direct = False
    doc1.theoretical = False
    
    # 2. An unstable candidate
    doc2 = MagicMock()
    doc2.material_id = "mp-2"
    doc2.formula_pretty = "LiFePO4" # Polymorph
    doc2.energy_above_hull = 0.15 # 150 meV (Unstable)
    doc2.is_stable = False
    doc2.band_gap = 3.0
    doc2.is_metal = False
    doc2.theoretical = True
    
    # 3. A metal (should be filtered if asking for semiconductors)
    doc3 = MagicMock()
    doc3.material_id = "mp-3"
    doc3.formula_pretty = "Fe"
    doc3.energy_above_hull = 0.0
    doc3.is_stable = True
    doc3.band_gap = 0.0
    doc3.is_metal = True
    doc3.theoretical = False

    return [doc1, doc2, doc3]


@pytest.fixture
def mock_entries():
    """
    Creates fake Phase Diagram entries.
    """
    entry1 = MagicMock()
    entry1.name = "Li"
    entry1.composition.reduced_formula = "Li"
    entry1.energy_per_atom = -1.9
    
    entry2 = MagicMock()
    entry2.name = "O2"
    entry2.composition.reduced_formula = "O2"
    entry2.energy_per_atom = -4.9
    
    return [entry1, entry2]


# =============================================================================
# ENVIRONMENT & CLEANUP FIXTURES
# =============================================================================

@pytest.fixture
def temp_dirs(tmp_path):
    """
    Overrides the global CACHE_DIR and OUTPUT_DIR for tests.
    This ensures tests don't write junk files to your actual project folders.
    """
    # Create temporary directories
    cache = tmp_path / "cache"
    outputs = tmp_path / "outputs"
    cache.mkdir()
    outputs.mkdir()
    
    # We yield the paths so tests can inspect them if needed
    yield {"cache": cache, "output": outputs}
    
    # Cleanup happens automatically by pytest's tmp_path, 
    # but we can do extra cleanup here if necessary.

