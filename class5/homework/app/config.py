"""
config.py
---------
Centralized configuration for paths, constants, and logging setup.

Responsibilities
----------------
    • Define base directories and file paths for data and models.
    • Store global constants like embedding model name and top-k value.
    • Configure structured logging for all modules.

Usage
-----
    from app.config import (
        BASE_DIR, DATA_DIR, FAISS_INDEX_PATH, CHUNK_ID_PATH,
        SQLITE_DB_PATH, LOG_PATH, EMBEDDING_MODEL, TOP_K, logger
    )

Author
------
    Carlos (refactored for clarity, documentation, and structured logging style)
"""

# =========================================================
# Imports and Paths
# =========================================================

import logging
import os

# Base directory: class5/homework
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Data directory: class5/homework/data
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)  # Ensure data folder exists

# File paths for index and metadata
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss_index.bin")
CHUNK_ID_PATH = os.path.join(DATA_DIR, "chunk_ids.pkl")
SQLITE_DB_PATH = os.path.join(DATA_DIR, "hybrid_index.db")
LOG_PATH = os.path.join(DATA_DIR, "retrieval_log.json")

# Embedding model and search configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 3  # Default number of top results to return

# =========================================================
# Logging Configuration
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

# Shared logger for all modules
logger = logging.getLogger("HybridSearch")
