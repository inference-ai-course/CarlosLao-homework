# voice_assistant/asr.py
import logging
import os
import tempfile
import wave
from contextlib import closing

import whisper

logger = logging.getLogger("voice_assistant")

class ASRService:
    def __init__(self, model_name: str = "small"):
        logger.info("Loading Whisper ASR model: %s", model_name)
        self.model = whisper.load_model(model_name)
        logger.info("Whisper loaded.")

    def transcribe(self, audio_path: str) -> str:
        logger.info("Transcribing: %s", audio_path)
        try:
            result = self.model.transcribe(audio_path, verbose=False)
            text = result.get("text", "").strip()
            logger.info("Transcription complete (chars=%d)", len(text))
            return text
        finally:
            try:
                os.remove(audio_path)
            except Exception:
                pass

    def ready(self) -> bool:
        path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                path = tmp.name
                with closing(wave.open(path, "w")) as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(16000)
                    wf.writeframes(b"\x00\x00" * 16000)
            self.model.transcribe(path, verbose=False)
            return True
        except Exception as e:
            logger.error("ASR readiness failed: %s", e)
            return False
        finally:
            if path:
                try:
                    os.remove(path)
                except Exception:
                    pass
