"""
run_indexing.py
---------------
Executes the indexing pipeline to populate SQLite and build FAISS index.

Responsibilities
----------------
    • Set up SQLite database and FTS5 table.
    • Insert sample document metadata and chunked text.
    • Generate FAISS index and save chunk IDs.

Usage
-----
    python run_indexing.py

Author
------
    Carlos (refactored for clarity, documentation, and structured logging style)
"""

# =========================================================
# Imports
# =========================================================

from app.config import logger
from app.indexing import (build_faiss_index, insert_chunk, insert_document,
                          setup_sqlite)

# =========================================================
# Execution Logic
# =========================================================

def run_indexing() -> None:
    """
    Executes the indexing pipeline with sample data.

    Returns
    -------
    None
    """
    logger.info("Starting indexing pipeline...")

    # Step 1: Set up SQLite and create necessary tables
    conn = setup_sqlite()

    # Step 2: Insert a sample document into the metadata table
    doc_id = "doc123"
    logger.info(f"Inserting document: {doc_id}")
    insert_document(
        conn,
        doc_id=doc_id,
        title="Transformer Paper",
        authors="Carlos Lao",
        year=2023,
        keywords="transformers, NLP"
    )

    # Step 3: Define and insert sample text chunks for the document
    chunks = [
        "Transformers use self-attention to process sequences.",
        "They are widely used in language modeling tasks.",
        "The encoder-decoder structure enables translation and summarization."
    ]

    chunk_ids = []
    for i, text in enumerate(chunks):
        chunk_id = f"{doc_id}_chunk{i+1}"
        insert_chunk(conn, chunk_id, doc_id, text)
        chunk_ids.append(chunk_id)
        logger.debug(f"Inserted chunk: {chunk_id}")

    # Step 4: Build FAISS index from the inserted chunks
    logger.info("Building FAISS index from inserted chunks...")
    build_faiss_index(chunks, chunk_ids)

    logger.info("✅ Indexing complete. FAISS and SQLite files created.")

# =========================================================
# Entry Point
# =========================================================

if __name__ == "__main__":
    run_indexing()
