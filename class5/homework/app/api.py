"""
api.py
------
Defines FastAPI endpoints for hybrid search.

Responsibilities
----------------
    • Expose a /hybrid_search endpoint for querying the index.
    • Format and return top-k results as JSON.
    • Log incoming requests and search outcomes.

Usage
-----
    from app.api import app

Author
------
    Carlos (refactored for clarity, documentation, and structured logging style)
"""

# =========================================================
# Imports
# =========================================================

from app.config import TOP_K, logger
from app.retrieval import get_chunk_text, hybrid_search
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse

# =========================================================
# FastAPI Application
# =========================================================

app = FastAPI()

# =========================================================
# Endpoint: /hybrid_search
# =========================================================

@app.get("/hybrid_search")
def hybrid_search_endpoint(q: str = Query(..., description="Search query"), k: int = TOP_K):
    """
    Handles hybrid search requests and returns top-k results.

    Parameters
    ----------
    q : str
        The user query string.
    k : int, optional
        Number of top results to return (default is 3).

    Returns
    -------
    JSONResponse
        A JSON object containing the query and ranked results.

    Raises
    ------
    HTTPException
        If the search fails due to internal error.
    """
    logger.info(f"Received /hybrid_search request: query='{q}', top_k={k}")
    try:
        # Run hybrid search pipeline
        results = hybrid_search(q, top_k=k)

        # Format results for JSON response
        response = []
        for chunk_id, score in results:
            text = get_chunk_text(chunk_id)
            response.append({
                "chunk_id": chunk_id,
                "score": round(score, 4),
                "text": text[:300] + "..." if len(text) > 300 else text
            })

        logger.info(f"Returning {len(response)} results for query: '{q}'")
        return JSONResponse(content={"query": q, "results": response})

    except Exception as e:
        logger.error(f"Hybrid search failed for query '{q}': {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
