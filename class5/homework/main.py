"""
main.py
-------
Launches the FastAPI application server.

Responsibilities
----------------
    • Starts the Uvicorn server with the configured FastAPI app.
    • Logs server startup and shutdown events.

Usage
-----
    python main.py

Author
------
    Carlos (refactored for clarity, documentation, and structured logging style)
"""

# =========================================================
# Imports
# =========================================================

import uvicorn
from app.api import app
from app.config import logger

# =========================================================
# Entry Point
# =========================================================

if __name__ == "__main__":
    # Log server startup
    logger.info("Starting FastAPI server...")

    # Launch the app using Uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

    # Log server shutdown (this line executes only if run blocking)
    logger.info("FastAPI server shutdown.")
