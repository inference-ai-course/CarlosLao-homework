# voice_assistant/config.py
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Config:
    HUGGINGFACE_TOKEN: str = os.getenv("HUGGINGFACE_TOKEN", "")
    TRANSCRIPT_FILE: str = os.getenv("TRANSCRIPT_FILE", "transcript.json")
    RESPONSE_FOLDER: str = os.getenv("RESPONSE_FOLDER", "response")
    PORT: int = int(os.getenv("PORT", "8000"))
    MAX_TURNS: int = int(os.getenv("MAX_TURNS", "5"))
    MODEL_NAME: str = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
    WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "small")

    def validate(self):
        if not self.HUGGINGFACE_TOKEN:
            raise RuntimeError("Missing Hugging Face token. Set HUGGINGFACE_TOKEN.")
