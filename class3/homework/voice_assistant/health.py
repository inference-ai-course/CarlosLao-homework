# voice_assistant/health.py
import logging
import subprocess

from .asr import ASRService
from .llm import LLMService
from .tts import TTSService

logger = logging.getLogger("voice_assistant")

def ffmpeg_installed() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except Exception as e:
        logger.error("FFmpeg check failed: %s", e)
        return False

def status(asr: ASRService | None, llm: LLMService | None, tts: TTSService | None) -> dict:
    return {
        "ffmpeg_installed": ffmpeg_installed(),
        "asr_model_loaded": asr.ready() if asr else False,
        "llm_ready": llm.ready() if llm else False,
        "gtts_ready": tts.ready() if tts else False,
    }
