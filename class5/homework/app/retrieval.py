# =========================================================
# Module Template
# =========================================================

"""
retrieval.py
------------
Provides semantic, keyword, and hybrid search capabilities.

Responsibilities
----------------
    • Perform FAISS-based semantic search.
    • Perform FTS5 or BM25 keyword search.
    • Merge results using Reciprocal Rank Fusion.
    • Retrieve chunk text from SQLite.

Usage
-----
    from app.retrieval import (
        semantic_search,
        keyword_search_fts5,
        keyword_search_bm25,
        hybrid_search,
        get_chunk_text
    )

Author
------
    Carlos (refactored for clarity, documentation, and structured logging style)
"""

# =========================================================
# Imports
# =========================================================

import pickle
import sqlite3

import faiss
import numpy as np
from app.config import (CHUNK_ID_PATH, EMBEDDING_MODEL, FAISS_INDEX_PATH,
                        SQLITE_DB_PATH, TOP_K, logger)
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# =========================================================
# Initialization
# =========================================================

model = SentenceTransformer(EMBEDDING_MODEL)
faiss_index = faiss.read_index(FAISS_INDEX_PATH)
with open(CHUNK_ID_PATH, "rb") as f:
    chunk_ids = pickle.load(f)

# =========================================================
# Search Functions
# =========================================================

def semantic_search(query: str, top_k: int = TOP_K) -> list[tuple[str, float]]:
    """
    Performs semantic search using FAISS index.

    Parameters
    ----------
    query : str
        User query string.
    top_k : int, optional
        Number of top results to return.

    Returns
    -------
    list of tuple
        List of (chunk_id, score) pairs.
    """
    logger.info(f"Semantic search initiated for query: '{query}'")
    embedding = model.encode([query]).astype("float32")
    D, I = faiss_index.search(embedding, top_k)
    return [(chunk_ids[i], float(D[0][j])) for j, i in enumerate(I[0])]


def keyword_search_fts5(query: str, top_k: int = TOP_K) -> list[tuple[str, float]]:
    """
    Performs keyword search using SQLite FTS5.

    Parameters
    ----------
    query : str
        User query string.
    top_k : int, optional
        Number of top results to return.

    Returns
    -------
    list of tuple
        List of (chunk_id, score) pairs. Score is fixed at 1.0.
    """
    logger.info(f"Keyword search (FTS5) initiated for query: '{query}'")
    with sqlite3.connect(SQLITE_DB_PATH) as conn:
        cursor = conn.cursor()

        # Wrap query in double quotes to prevent column misinterpretation
        safe_query = f'"{query}"'

        cursor.execute("""
            SELECT chunk_id, chunk_text FROM doc_chunks
            WHERE doc_chunks MATCH ?
            LIMIT ?
        """, (safe_query, top_k))

        results = [(row[0], 1.0) for row in cursor.fetchall()]
        logger.debug(f"Keyword search returned {len(results)} results.")
        return results


def keyword_search_bm25(query: str, top_k: int = TOP_K) -> list[tuple[str, float]]:
    """
    Performs keyword search using BM25 ranking.

    Parameters
    ----------
    query : str
        User query string.
    top_k : int, optional
        Number of top results to return.

    Returns
    -------
    list of tuple
        List of (chunk_id, score) pairs.
    """
    logger.info(f"Keyword search (BM25) initiated for query: '{query}'")
    with sqlite3.connect(SQLITE_DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT chunk_id, chunk_text FROM doc_chunks")
        rows = cursor.fetchall()

        # Prepare corpus and IDs
        corpus = [r[1].split() for r in rows]
        ids = [r[0] for r in rows]

        # Compute BM25 scores
        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores(query.split())
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = [(ids[i], scores[i]) for i in top_indices]
        logger.debug(f"BM25 search returned {len(results)} results.")
        return results


def reciprocal_rank_fusion(results_a: list[tuple[str, float]], results_b: list[tuple[str, float]], k: int = 60) -> list[tuple[str, float]]:
    """
    Merges two ranked result lists using Reciprocal Rank Fusion.

    Parameters
    ----------
    results_a : list of tuple
        First result list (e.g., semantic).
    results_b : list of tuple
        Second result list (e.g., keyword).
    k : int, optional
        RRF constant to control score decay.

    Returns
    -------
    list of tuple
        Combined and re-ranked list of (chunk_id, fused_score).
    """
    logger.debug("Applying Reciprocal Rank Fusion...")
    scores = {}

    # Score from first list
    for rank, (chunk_id, _) in enumerate(results_a):
        scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank)

    # Score from second list
    for rank, (chunk_id, _) in enumerate(results_b):
        scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank)

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    logger.debug(f"RRF produced {len(fused)} fused results.")
    return fused


def hybrid_search(query: str, top_k: int = TOP_K, use_bm25: bool = False) -> list[tuple[str, float]]:
    """
    Performs hybrid search by combining semantic and keyword results.

    Parameters
    ----------
    query : str
        User query string.
    top_k : int, optional
        Number of top results to return.
    use_bm25 : bool, optional
        Whether to use BM25 instead of FTS5 for keyword search.

    Returns
    -------
    list of tuple
        Final ranked list of (chunk_id, fused_score).
    """
    logger.info(f"Hybrid search initiated for query: '{query}'")
    semantic = semantic_search(query, top_k)
    keyword = keyword_search_bm25(query, top_k) if use_bm25 else keyword_search_fts5(query, top_k)
    fused = reciprocal_rank_fusion(semantic, keyword)
    logger.info(f"Hybrid search completed with {len(fused[:top_k])} results.")
    return fused[:top_k]


def get_chunk_text(chunk_id: str) -> str:
    """
    Retrieves the full text of a chunk from the SQLite database.

    Parameters
    ----------
    chunk_id : str
        Unique identifier of the chunk.

    Returns
    -------
    str
        Text content of the chunk. Returns an empty string if not found.
    """
    logger.debug(f"Retrieving text for chunk_id: {chunk_id}")
    with sqlite3.connect(SQLITE_DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT chunk_text FROM doc_chunks WHERE chunk_id = ?", (chunk_id,))
        row = cursor.fetchone()
        return row[0] if row else ""
