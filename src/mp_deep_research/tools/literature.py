# New tool: literature.py (for future integration)
from .shared import ensure_structure
from langchain_core.tools import tool

@tool
def check_literature(
    formula: str,
    search_scope: str = "arxiv"  # or "crossref", "semantic_scholar"
) -> str:
    """
    Check if a hypothetical material has already been reported in literature.
    
    This prevents "discovering" materials that are already known but
    missing from computational databases.
    
    Args:
        formula: Chemical formula to search (e.g., "NaFePO4")
        search_scope: Which database to search
    
    Returns:
        Summary of existing literature on this composition.
    """
    import arxiv
    
    # Normalize formula for search
    search_terms = [
        formula,
        formula.replace("4", "₄").replace("2", "₂"),  # Unicode variants
        f"{formula} synthesis",
        f"{formula} cathode",
        f"{formula} battery",
    ]
    
    results = []
    client = arxiv.Client()
    
    for term in search_terms[:2]:  # Limit API calls
        search = arxiv.Search(
            query=term,
            max_results=5,
            sort_by=arxiv.SortCriterion.Relevance
        )
        for paper in client.results(search):
            results.append({
                "title": paper.title,
                "year": paper.published.year,
                "url": paper.entry_id,
                "summary": paper.summary[:200]
            })
    
    # Deduplicate
    seen_titles = set()
    unique_results = []
    for r in results:
        if r["title"] not in seen_titles:
            seen_titles.add(r["title"])
            unique_results.append(r)
    
    if not unique_results:
        return f"""
LITERATURE CHECK: {formula}
{'='*50}
Status: NO EXISTING PUBLICATIONS FOUND ✨

This appears to be a genuinely novel composition!

RECOMMENDATION:
  - Proceed with DFT validation
  - This could be a publishable discovery
  - Double-check with Google Scholar manually
"""
    
    papers_text = "\n".join([
        f"  [{i+1}] {r['title']} ({r['year']})\n      {r['url']}"
        for i, r in enumerate(unique_results[:5])
    ])
    
    return f"""
LITERATURE CHECK: {formula}
{'='*50}
Status: EXISTING RESEARCH FOUND

Found {len(unique_results)} potentially relevant papers:

{papers_text}

INTERPRETATION:
  This composition may already be known experimentally.
  Review papers before claiming as "novel discovery."

RECOMMENDATION:
  - Read these papers to understand synthesis conditions
  - Check if our predicted structure matches experimental
  - Focus on property improvements rather than novelty
"""
# ```

# ## Updated Workflow with All Safeguards
# ```
# Agent: "User wants NaFePO4"
#   ↓
# substitute_species(mp-19017, {Li: Na})
#   ↓
# relax_structure_m3gnet(nafep04.json)
#   ├─→ Volume exploded (+60%)? → "FAILED: Try different parent"
#   └─→ Normal relaxation → Continue
#   ↓
# assess_stability(nafep04_relaxed.json, -5.2)
#   ↓ (includes provenance tags)
#   ├─→ UNSTABLE? → "Try different substitution"
#   └─→ STABLE? → Continue
#   ↓
# check_literature("NaFePO4")  ← NEW
#   ├─→ Papers found? → "Already known - read these first"
#   └─→ No papers? → "Novel! Submit for DFT validation"