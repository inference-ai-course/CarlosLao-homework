import asyncio
import os
import utils
from contextlib import asynccontextmanager
from fastapi import FastAPI

from voice_assistant.config import Config
from voice_assistant.logging_config import setup_logging
from voice_assistant.utils import ensure_dir
from voice_assistant.asr import ASRService
from voice_assistant.llm import LLMService
from voice_assistant.tts import TTSService
from voice_assistant.memory import Memory
from voice_assistant.routers.chat import init_router as chat_router
from voice_assistant.routers.memory import init_router as memory_router
from voice_assistant.routers.health import init_router as health_router

config = Config()
logger = setup_logging()

# Shared singletons
asr_service: ASRService | None = None
llm_service: LLMService | None = None
tts_service: TTSService | None = None
memory = Memory(max_turns=config.MAX_TURNS)

app = FastAPI(title="Voice Assistant", version="3.0")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing application resources...")
    ensure_dir(config.RESPONSE_FOLDER)

    asr_service, llm_service, tts_service = await asyncio.gather(
        asyncio.to_thread(ASRService, config.WHISPER_MODEL),
        asyncio.to_thread(LLMService, config.MODEL_NAME, config.HUGGINGFACE_TOKEN),
        asyncio.to_thread(TTSService, "en", config.RESPONSE_FOLDER),
    )

    memory = Memory()

    logger.info("ASR, LLM, and TTS services initialized.")

    # Attach services to app.state for shared access
    app.state.config = config
    app.state.asr = asr_service
    app.state.llm = llm_service
    app.state.tts = tts_service
    app.state.memory = memory

    # Dynamically include routers after everything’s ready
    app.include_router(chat_router(config, asr_service, llm_service, tts_service, memory), prefix="/chat")
    app.include_router(memory_router(memory, config), prefix="/memory")
    app.include_router(health_router(config, lambda: (asr_service, llm_service, tts_service)), prefix="/health")

    yield
    logger.info("Application shutdown complete.")

# Apply lifespan AFTER definition
app.router.lifespan_context = lifespan

if __name__ == "__main__":
    utils.add_root_dir()
    import uvicorn
    port = int(os.getenv("PORT", str(config.PORT)))
    logger.info("Starting server on 0.0.0.0:%d", port)
    uvicorn.run(app, host="0.0.0.0", port=port)
