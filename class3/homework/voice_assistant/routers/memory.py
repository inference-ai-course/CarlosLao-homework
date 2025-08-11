# voice_assistant/routers/memory.py
import logging
import os
from fastapi import APIRouter

from ..config import Config
from ..memory import Memory

logger = logging.getLogger("voice_assistant")

def init_router(memory: Memory, config: Config) -> APIRouter:
    router = APIRouter()

    @router.get("/")
    async def get_memory():
        snapshot = await memory.snapshot()
        logger.info("Returned memory snapshot (%d messages)", len(snapshot))
        return snapshot

    @router.delete("/")
    async def clear_memory():
        n = await memory.clear()
        logger.info("Cleared memory (removed %d messages)", n)
        try:
            if os.path.exists(config.TRANSCRIPT_FILE):
                os.remove(config.TRANSCRIPT_FILE)
                logger.info("Deleted transcript file: %s", config.TRANSCRIPT_FILE)
        except Exception as e:
            logger.warning("Failed to delete transcript file: %s", e)
        return {"cleared": True}

    return router
