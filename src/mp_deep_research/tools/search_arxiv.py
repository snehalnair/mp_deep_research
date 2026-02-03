"""
Literature search tools for finding scientific context.
"""
from typing import List, Optional
from langchain_core.tools import tool
from pydantic import BaseModel, Field
import arxiv

# =============================================================================
# INPUT SCHEMA
# =============================================================================

class ArxivSearchSchema(BaseModel):
    query: str = Field(description="Search query (e.g., 'LiFePO4 cathode stability', 'solid state electrolyte')")
    max_results: int = Field(default=5, description="Maximum number of papers to return (default: 5)")
    sort_by: str = Field(default="relevance", description="Sort order: 'relevance' or 'lastUpdatedDate'")

# =============================================================================
# TOOL: SEARCH ARXIV
# =============================================================================

@tool(args_schema=ArxivSearchSchema)
def search_arxiv(
    query: str,
    max_results: int = 5,
    sort_by: str = "relevance"
) -> str:
    """
    Search arXiv for scientific papers to validate findings or gather context.
    
    Use this tool to:
    - Check if a "new" material has already been synthesized.
    - Find synthesis recipes for similar materials.
    - Understand known stability issues (e.g., "why does LiMnPO4 fail?").
    
    Args:
        query: Search keywords.
        max_results: Limit results (default 5).
        sort_by: 'relevance' (default) or 'lastUpdatedDate' (for newest papers).
        
    Returns:
        A formatted summary of relevant papers including titles, authors, and abstracts.
    """
    try:
        # Map string sort criteria to arxiv.SortCriterion
        sort_map = {
            "relevance": arxiv.SortCriterion.Relevance,
            "lastUpdatedDate": arxiv.SortCriterion.LastUpdatedDate,
            "submittedDate": arxiv.SortCriterion.SubmittedDate,
        }
        criterion = sort_map.get(sort_by, arxiv.SortCriterion.Relevance)

        # Construct Client (more robust than basic Search)
        client = arxiv.Client()
        
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=criterion
        )

        results = []
        for result in client.results(search):
            # Clean up the summary (remove newlines)
            summary_clean = result.summary.replace("\n", " ")
            
            results.append(f"""
PAPER: {result.title}
DATE: {result.published.strftime('%Y-%m-%d')}
AUTHORS: {', '.join(a.name for a in result.authors[:3])}{' et al.' if len(result.authors) > 3 else ''}
URL: {result.entry_id}
SUMMARY: {summary_clean[:400]}...
""")

        if not results:
            return f"No papers found for query: '{query}'"

        return f"""
LITERATURE SEARCH RESULTS for '{query}'
{'='*50}
{ "".join(results) }
{'='*50}
"""

    except Exception as e:
        return f"ArXiv search failed: {e}"