"""
Unit tests for innovation.py tools.
"""
import pytest
from unittest.mock import MagicMock, patch
from pymatgen.core import Structure, Lattice

# Explicitly import mp_api.client
import mp_api.client 

from mp_deep_research.tools.innovation import substitute_species, relax_structure_m3gnet

@pytest.fixture
def simple_structure():
    """Creates a simple Li2O structure for testing."""
    return Structure(
        Lattice.cubic(4.0),
        ["Li", "Li", "O"],
        [[0, 0, 0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25]]
    )

def test_substitute_species_logic(simple_structure, tmp_path):
    """Test that atomic substitution works correctly."""
    
    # This patch works only if MPRester is imported INSIDE the tool function
    with patch("mp_api.client.MPRester") as MockRester:
        # Setup the mock to return our fake structure
        mock_mpr = MockRester.return_value.__enter__.return_value
        mock_mpr.get_structure_by_material_id.return_value = simple_structure
        
        with patch("mp_deep_research.tools.innovation.OUTPUT_DIR", tmp_path):
            result = substitute_species.invoke({
                "material_id": "mp-123", 
                "substitutions": {"Li": "Na"}
            })
            
            # Now this assertion should pass because it used the mock
            assert "SUBSTITUTION SUCCESSFUL" in result
            assert "Na2O" in result.replace(" ", "")

@patch("mp_deep_research.tools.innovation.HAS_MATGL", True)
@patch("mp_deep_research.tools.innovation.matgl")
@patch("mp_deep_research.tools.innovation.Relaxer")
def test_relax_structure_m3gnet_mock(mock_relaxer_class, mock_matgl, simple_structure, tmp_path):
    """Test the M3GNet workflow with Uncertainty Quantification."""
    
    mock_instance = mock_relaxer_class.return_value
    
    # Mock relaxed structure
    relaxed_struct = simple_structure.copy()
    relaxed_struct.scale_lattice(simple_structure.volume * 0.95)
    
    # We expect 3 calls (1 standard + 2 perturbed). 
    # Let's return slightly different energies to test std_dev calculation
    mock_instance.relax.side_effect = [
        {"final_structure": relaxed_struct, "trajectory": MagicMock(energies=[-10.0])}, # Run 1
        {"final_structure": relaxed_struct, "trajectory": MagicMock(energies=[-10.1])}, # Run 2
        {"final_structure": relaxed_struct, "trajectory": MagicMock(energies=[-9.9])},  # Run 3
    ]
    
    # Save dummy input
    input_path = tmp_path / "input.json"
    simple_structure.to(filename=str(input_path))
    
    # Run
    result = relax_structure_m3gnet.invoke({
        "structure_path": str(input_path),
        "estimate_uncertainty": True
    })
    
    # Verify
    assert "SIMULATION RESULTS" in result
    assert "+/-" in result # Check if error bar is present
    assert "Confidence:" in result
    
    # Check if relax was called 3 times
    assert mock_instance.relax.call_count == 3


# ... (Previous imports) ...
from pymatgen.analysis.phase_diagram import PDEntry
from pymatgen.core import Composition
from mp_deep_research.tools.innovation import assess_stability # Import new tool

def test_assess_stability_logic():
    """Test the convex hull calculation logic."""
    
    # 1. Setup Mock Entries using REAL elements (Li, O) to satisfy Pymatgen validation
    # We will pretend Li-O system only has two stable phases: Li and O.
    entry_Li = PDEntry(Composition("Li"), -1.0) # Pure Li is -1.0 eV/atom
    entry_O = PDEntry(Composition("O"), -1.0)   # Pure O is -1.0 eV/atom
    
    # 2. Mock MPRester
    with patch("mp_api.client.MPRester") as MockRester:
        mock_mpr = MockRester.return_value.__enter__.return_value
        # Return our fake universe where only Li and O exist
        mock_mpr.get_entries_in_chemsys.return_value = [entry_Li, entry_O]
        
        # 3. Test a STABLE candidate (LiO at -1.5 eV/atom)
        # The hull connects Li(-1.0) and O(-1.0). Average is -1.0.
        # Our candidate is -1.5, which is LOWER than the hull -> Stable.
        result_stable = assess_stability.invoke({
            "composition": "LiO", 
            "energy_per_atom": -1.5
        })
        
        assert "Is Stable? YES" in result_stable
        assert "Energy Above Hull: 0.0000" in result_stable
        
        # 4. Test an UNSTABLE candidate (LiO at -0.5 eV/atom)
        # -0.5 is HIGHER than the hull (-1.0) -> Unstable.
        result_unstable = assess_stability.invoke({
            "composition": "LiO", 
            "energy_per_atom": -0.5
        })
        
        assert "Is Stable? NO" in result_unstable
        assert "Decomposes into" in result_unstable