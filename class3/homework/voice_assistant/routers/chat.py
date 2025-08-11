# voice_assistant/routers/chat.py
import asyncio
import logging
import utils
from typing import Callable

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from ..config import Config
from ..asr import ASRService
from ..llm import LLMService
from ..tts import TTSService
from ..memory import Memory
from ..prompt import format_prompt
from ..utils import save_json, save_upload_to_temp

logger = logging.getLogger("voice_assistant")

ACCEPTED_AUDIO_TYPES = {
    "audio/mpeg", "audio/wav", "audio/x-wav", "audio/x-m4a", "audio/ogg", "audio/webm", "application/octet-stream"
}

def init_router(
    config: Config,
    asr: ASRService | None,
    llm: LLMService | None,
    tts: TTSService | None,
    memory: Memory
) -> APIRouter:
    router = APIRouter()

    def ensure_ready():
        if not asr or not llm or not tts:
            raise HTTPException(503, "Services not initialized")
        # soft readiness: verify subservices quickly
        if not (llm.ready() and tts.ready()):
            raise HTTPException(500, "Model services not ready")

    @router.post("/")
    async def chat(file: UploadFile = File(...)):
        logger.info("Received /chat request")
        ensure_ready()

        # Save upload and transcribe
        try:
            temp_path = await save_upload_to_temp(file, ACCEPTED_AUDIO_TYPES)
        except ValueError as ve:
            raise HTTPException(400, str(ve))

        user_text = await asyncio.to_thread(asr.transcribe, temp_path)  # type: ignore[arg-type]
        await memory.append("user", user_text)

        # Build prompt and generate
        history = await memory.snapshot()
        prompt = format_prompt(user_text, history)
        assistant_reply = await asyncio.to_thread(llm.generate, prompt)  # type: ignore[arg-type]
        await memory.append("assistant", assistant_reply)

        # Persist transcript
        snapshot = await memory.snapshot()
        transcript_path = utils.get_file_path(config.TRANSCRIPT_FILE)
        await asyncio.to_thread(save_json, transcript_path, snapshot)

        # Synthesize TTS
        tts_path = await asyncio.to_thread(tts.synthesize, assistant_reply, file.filename)  # type: ignore[arg-type]

        logger.info("Completed /chat request")
        return JSONResponse(
            content={
                "transcription": user_text,
                "response": assistant_reply,
                "tts_audio": tts_path,
                "conversation_memory": snapshot,
                "debug_prompt": prompt,
            }
        )

    return router
