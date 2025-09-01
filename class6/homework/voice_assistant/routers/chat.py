"""
routers/chat.py
---------------
FastAPI router for the /chat endpoint.

Handles audio upload from the client, performs ASR, routes the text to the
language model, synthesizes a TTS reply, and returns all relevant metadata.
"""

import os
import asyncio
import tempfile
import logging
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
from ..config import ACCEPTED_AUDIO_TYPES, RESPONSE_FOLDER
from ..utils import get_file_path, guess_media_type
from ..prompt import format_prompt
from ..memory import MemoryService
from ..asr import ASRService
from ..llm import LLMService
from ..tts import TTSService

logger = logging.getLogger(__name__)
router = APIRouter()


class Services:
    """
    Aggregate container for all assistant services used in this route.
    """
    def __init__(self, asr, llm, tts, memory):
        self.asr = asr
        self.llm = llm
        self.tts = tts
        self.memory = memory


def get_services(req: Request) -> Services:
    """
    Dependency injection hook to access the service bundle from app state.

    Using a typed Request ensures FastAPI treats this as an internal dependency
    (not a client-supplied 'request' parameter in the docs).
    """
    return req.app.state.services


async def save_temp_audio(file: UploadFile) -> str:
    """
    Save uploaded audio to a temporary file and return the path.

    Chooses a suffix from the incoming filename; falls back to .bin if missing.
    """
    suffix = os.path.splitext(file.filename or "")[1] or ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        data = await file.read()
        tmp.write(data)
        return tmp.name


@router.post("/chat/")
async def chat(
    file: UploadFile = File(...),
    services: Services = Depends(get_services),
):
    """
    Primary chat endpoint.

    1. Validate uploaded audio.
    2. Transcribe audio to text (ASR).
    3. Update memory with user's message.
    4. Generate assistant's reply (LLM).
    5. Convert reply to audio (TTS).
    6. Return JSON containing all relevant data.
    """
    # Effective content type: prefer client-sent; fall back to guess by filename.
    effective_ctype = file.content_type or guess_media_type(file.filename or "")
    logger.info("Incoming upload: filename=%s content_type=%s", file.filename, effective_ctype)

    if effective_ctype not in ACCEPTED_AUDIO_TYPES:
        raise HTTPException(status_code=415, detail=f"Unsupported media type: {effective_ctype}")

    # Save and transcribe audio
    audio_path = await save_temp_audio(file)
    try:
        user_text = await asyncio.to_thread(services.asr.transcribe, audio_path)
    finally:
        try:
            os.remove(audio_path)
        except Exception:
            pass

    # Conversation and response generation
    await services.memory.add_user(user_text, "user_1")  # Simplified speaker_id
    history = await services.memory.get_recent()
    logger.info(history)
    prompt = format_prompt(history)
    reply = await asyncio.to_thread(services.llm.generate, prompt)

    # Synthesize TTS to mounted /audio directory (TTSService returns a URL path)
    tts_url = await asyncio.to_thread(
        services.tts.synthesize, reply, file.filename or "response", "user_1"
    )

    logger.info("Prompt: %s", prompt)
    logger.info("Reply: %s", reply)
    await services.memory.add_assistant(reply)
    await services.memory.save_transcript()

    return JSONResponse(
        content={
            "speaker_id": "user_1",
            "transcription": user_text,
            "response": reply,
            "tts_audio": tts_url,
            "conversation_memory": history,
            "debug_prompt": prompt,
        }
    )
