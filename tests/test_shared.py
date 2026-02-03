import pytest
from unittest.mock import patch, MagicMock
from mp_deep_research.tools.shared import query_mp_with_resilience

@patch("mp_deep_research.tools.shared.MPRester")
def test_api_fallback_logic(mock_mprester_class):
    # Setup: Create a mock client instance
    mock_client = mock_mprester_class.return_value.__enter__.return_value
    
    # Scenario: First call fails with a "field not found" error
    # This simulates MP API removing a field we asked for
    mock_client.materials.summary.search.side_effect = [
        Exception("Field 'volume' not recognized in schema"), # First attempt
        [MagicMock(material_id="mp-123")]                     # Second attempt (success)
    ]

    # Run the function
    docs, raw_data = query_mp_with_resilience("Li-Fe-O")

    # Assertions
    # We expect it to have called search TWICE (once normal, once fallback)
    assert mock_client.materials.summary.search.call_count == 2
    assert docs[0].material_id == "mp-123"