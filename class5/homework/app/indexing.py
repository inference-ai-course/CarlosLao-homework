"""
indexing.py
-----------
Builds and populates the hybrid search index using SQLite and FAISS.

Responsibilities
----------------
    • Set up SQLite database with metadata and chunk storage.
    • Insert document metadata and chunked text into FTS5 table.
    • Generate and store FAISS embeddings for semantic search.

Usage
-----
    from app.indexing import (
        setup_sqlite,
        insert_document,
        insert_chunk,
        build_faiss_index
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
                        SQLITE_DB_PATH, logger)
from sentence_transformers import SentenceTransformer

# =========================================================
# Functions
# =========================================================

def setup_sqlite() -> sqlite3.Connection:
    """
    Initializes the SQLite database and creates required tables.

    Returns
    -------
    sqlite3.Connection
        Active connection to the SQLite database.
    """
    logger.info("Setting up SQLite database...")
    conn = sqlite3.connect(SQLITE_DB_PATH)
    cursor = conn.cursor()

    # Create documents table for metadata
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            doc_id TEXT PRIMARY KEY,
            title TEXT,
            authors TEXT,
            year INTEGER,
            keywords TEXT
        )
    """)

    # Create FTS5 table for chunked text
    cursor.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS doc_chunks USING fts5(
            chunk_id,
            doc_id,
            chunk_text
        )
    """)

    conn.commit()
    logger.info("SQLite tables created successfully.")
    return conn


def insert_document(conn: sqlite3.Connection, doc_id: str, title: str, authors: str, year: int, keywords: str) -> None:
    """
    Inserts a document's metadata into the SQLite database.

    Parameters
    ----------
    conn : sqlite3.Connection
        Active database connection.
    doc_id : str
        Unique identifier for the document.
    title : str
        Title of the document.
    authors : str
        Author(s) of the document.
    year : int
        Publication year.
    keywords : str
        Comma-separated keywords.

    Returns
    -------
    None
    """
    logger.debug(f"Inserting document metadata: {doc_id}")
    conn.execute("""
        INSERT OR REPLACE INTO documents VALUES (?, ?, ?, ?, ?)
    """, (doc_id, title, authors, year, keywords))
    conn.commit()


def insert_chunk(conn: sqlite3.Connection, chunk_id: str, doc_id: str, chunk_text: str) -> None:
    """
    Inserts a text chunk into the FTS5-enabled chunk table.

    Parameters
    ----------
    conn : sqlite3.Connection
        Active database connection.
    chunk_id : str
        Unique identifier for the chunk.
    doc_id : str
        ID of the parent document.
    chunk_text : str
        Text content of the chunk.

    Returns
    -------
    None
    """
    logger.debug(f"Inserting chunk: {chunk_id} (doc: {doc_id})")
    conn.execute("""
        INSERT INTO doc_chunks VALUES (?, ?, ?)
    """, (chunk_id, doc_id, chunk_text))
    conn.commit()


def build_faiss_index(chunk_texts: list[str], chunk_ids: list[str]) -> None:
    """
    Builds a FAISS index from chunk embeddings and saves it to disk.

    Parameters
    ----------
    chunk_texts : list of str
        List of chunk text strings.
    chunk_ids : list of str
        Corresponding chunk identifiers.

    Returns
    -------
    None
    """
    logger.info(f"Building FAISS index with {len(chunk_texts)} chunks...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Generate embeddings for each chunk
    embeddings = model.encode(chunk_texts, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    # Build and save FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, FAISS_INDEX_PATH)

    # Save chunk ID mapping
    with open(CHUNK_ID_PATH, "wb") as f:
        pickle.dump(chunk_ids, f)

    logger.info("FAISS index built and saved successfully.")
