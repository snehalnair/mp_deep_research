'''
module: tests/test_screening.py
'''
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mp_deep_research.tools.screening import batch_screen_materials  # type: ignore[import-not-found]

# 1. Create fake material objects (simulating MP API response)
@pytest.fixture
def mock_materials_fixture():
    '''docstring: mock_materials
    '''
    # Stable material
    mat1 = MagicMock()
    mat1.material_id = "mp-1"
    mat1.energy_above_hull = 0.0
    mat1.band_gap = 1.5
    mat1.is_metal = False
    
    # Unstable material
    mat2 = MagicMock()
    mat2.material_id = "mp-2"
    mat2.energy_above_hull = 0.1  # High energy (unstable)
    mat2.band_gap = 0.0
    mat2.is_metal = True
    
    return [mat1, mat2]

# 2. Test the function
@patch("mp_deep_research.tools.screening.query_mp_with_resilience")
@patch("mp_deep_research.tools.screening.save_to_cache") # Don't write real files
@patch("mp_deep_research.tools.screening.load_from_cache", return_value=None) # Force fresh query
def test_screening_filters_unstable(_mock_load, _mock_save, mock_query, mock_materials_fixture):
    '''docstring: test_screening_filters_unstable
    '''
    
    # Setup the mock to return our fake data
    mock_query.return_value = (mock_materials_fixture, []) # (docs, raw_data)

    # Run the tool
    result = batch_screen_materials.invoke(
        {
            "chemsys": "Li-Fe-O",
            "stability_threshold": 0.05,
        }
    )

    # Assertions
    assert "mp-1" in result
    assert "mp-2" not in result  # Should be filtered out
    assert "Total materials queried: 2" in result
    assert "Passed all filters: 1" in result