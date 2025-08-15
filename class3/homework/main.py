"""
main.py
-------
Primary entry point for the Voice Assistant FastAPI application.

This module is responsible for:

1. Initializing core services (ASR, LLM, TTS, Memory).
2. Configuring application lifespan events so services are loaded at startup
   and shared via dependency injection.
3. Registering routers for chat, memory, and health endpoints.
4. Mounting the static frontend from the /web directory.
5. Applying CORS middleware for cross‑origin requests.

The application can be launched with Uvicorn for development or production.
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Configuration and utility imports
from voice_assistant.config import (
    HUGGINGFACE_TOKEN,
    LLM_MODEL_ID,
    ASR_MODEL_NAME,
    WEB_DIR,
    RESPONSE_FOLDER,
)
from voice_assistant.utils import get_file_path, ensure_dir

# Service imports
from voice_assistant.asr import ASRService
from voice_assistant.llm import LLMService
from voice_assistant.tts import TTSService
from voice_assistant.memory import MemoryService

# Router imports
from voice_assistant.routers import chat, memory, health

# Configure logging format and level for the entire application
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("voice-assistant")


class AppServices:
    """
    Aggregate container for all core application services.
    Instances of this class are attached to FastAPI's state for access
    within routes and dependencies.
    """
    def __init__(self):
        self.asr = ASRService(ASR_MODEL_NAME)
        self.llm = LLMService(LLM_MODEL_ID, HUGGINGFACE_TOKEN)
        self.tts = TTSService()
        self.memory = MemoryService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for application startup and shutdown.

    On startup:
        - Instantiate and load ASR, LLM, and TTS services.
        - Ensure required directories exist (e.g., response/, web/).

    On shutdown:
        - Any necessary cleanup can be performed here (currently none).
    """
    logger.info("Initializing application services...")
    services = AppServices()

    # Load models and services into memory
    services.asr.load()
    services.llm.load()
    services.tts.load()

    # Ensure critical directories exist
    ensure_dir(get_file_path("", RESPONSE_FOLDER))
    ensure_dir(get_file_path("", WEB_DIR))

    # Attach to application state
    app.state.services = services

    logger.info("Application startup complete.")
    yield
    logger.info("Application shutdown complete.")


def create_app() -> FastAPI:
    """
    Factory function to create and configure the FastAPI application instance.

    Returns
    -------
    FastAPI
        The configured FastAPI application.
    """
    app = FastAPI(
        title="Voice Assistant",
        version="1.0",
        lifespan=lifespan
    )

    # Configure CORS to allow all origins (adjust as needed for security)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount static web frontend at /ui
    web_path = get_file_path("", WEB_DIR)
    audio_dir = get_file_path("", "audio")
    app.mount("/web", StaticFiles(directory=web_path, html=True), name="web")
    app.mount("/audio", StaticFiles(directory=audio_dir, html=True), name="audio")

    # Register API routers
    app.include_router(chat.router)
    app.include_router(memory.router)
    app.include_router(health.router)

    return app


# Application instance for ASGI servers (Uvicorn, Hypercorn, etc.)
app = create_app()

if __name__ == "__main__":
    """
    Allow running the application directly with:
        python main.py
    This uses Uvicorn's development server.
    """
    import uvicorn
    port = 8000
    logger.info("Starting development server at http://0.0.0.0:%d", port)
    uvicorn.run(app, host="0.0.0.0", port=port)
