"""
Unit tests for search_arxiv.py
"""
import pytest
from unittest.mock import MagicMock, patch
import datetime

# Import the tool
from mp_deep_research.tools.search_arxiv import search_arxiv

@pytest.fixture
def mock_arxiv_result():
    """Creates a fake arXiv result object."""
    mock_result = MagicMock()
    mock_result.title = "Deep Learning for Materials Discovery"
    mock_result.published = datetime.datetime(2023, 1, 1)
    
    # --- FIX START ---
    # Create author mocks and explicitly set their .name attribute to strings
    author1 = MagicMock()
    author1.name = "Alice Scientist"
    author2 = MagicMock()
    author2.name = "Bob Engineer"
    
    mock_result.authors = [author1, author2]
    # --- FIX END ---
    
    mock_result.entry_id = "http://arxiv.org/abs/2301.00000"
    mock_result.summary = "We present a new method for discovering materials using graph neural networks."
    return mock_result

def test_search_arxiv_success(mock_arxiv_result):
    """Test that search returns formatted string on success."""
    
    # Mock the Client and its results iterator
    with patch("arxiv.Client") as MockClient:
        instance = MockClient.return_value
        instance.results.return_value = [mock_arxiv_result]
        
        # Run tool
        result = search_arxiv.invoke({"query": "materials AI"})
        
        # Verify output
        assert "LITERATURE SEARCH RESULTS" in result
        assert "Deep Learning for Materials Discovery" in result
        assert "Alice Scientist" in result
        assert "http://arxiv.org/abs/2301.00000" in result

def test_search_arxiv_empty():
    """Test handling of no results."""
    
    with patch("arxiv.Client") as MockClient:
        instance = MockClient.return_value
        instance.results.return_value = []
        
        result = search_arxiv.invoke({"query": "nonexistent_material_xyz_123"})
        
        assert "No papers found" in result

def test_search_arxiv_error():
    """Test graceful error handling."""
    
    with patch("arxiv.Client") as MockClient:
        instance = MockClient.return_value
        instance.results.side_effect = Exception("API Connection Error")
        
        result = search_arxiv.invoke({"query": "crash me"})
        
        assert "ArXiv search failed" in result
        assert "API Connection Error" in result