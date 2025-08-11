# voice_assistant/routers/health.py
import logging
from fastapi import APIRouter

from ..config import Config
from ..health import status

logger = logging.getLogger("voice_assistant")

def init_router(config: Config, get_services) -> APIRouter:
    """
    get_services: callable returning (asr, llm, tts)
    """
    router = APIRouter()

    @router.get("/")
    def health():
        asr, llm, tts = get_services()
        s = status(asr, llm, tts)
        logger.info("Health status: %s", s)
        return s

    return router
