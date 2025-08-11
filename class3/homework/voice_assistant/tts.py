# voice_assistant/tts.py
import logging
import os
import tempfile
from gtts import gTTS

from utils import get_file_path

logger = logging.getLogger("voice_assistant")

class TTSService:
    def __init__(self, lang: str = "en", out_dir: str = "response"):
        self.lang = lang
        self.out_dir = out_dir

    def synthesize(self, text: str, input_filename: str) -> str:
        base_name = os.path.splitext(os.path.basename(input_filename))[0] or "response"
        out_path = get_file_path(f"{base_name}_response.mp3", self.out_dir)
        logger.info("Synthesizing TTS -> %s", out_path)
        gTTS(text=text, lang=self.lang).save(out_path)
        return out_path

    def ready(self) -> bool:
        path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                path = tmp.name
            gTTS("OK", lang=self.lang).save(path)
            return os.path.exists(path) and os.path.getsize(path) > 0
        except Exception as e:
            logger.error("gTTS readiness failed: %s", e)
            return False
        finally:
            if path:
                try:
                    os.remove(path)
                except Exception:
                    pass
