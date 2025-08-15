"""
tts.py
------
Text-to-Speech (TTS) module.

Provides the TTSService class to synthesize spoken audio from text
using either Google Text-to-Speech (gTTS) or a third-party API (e.g., ElevenLabs).
"""

import os
import time
import uuid
import json
from gtts import gTTS
from typing import Dict
from .config import ELEVENLABS_API_KEY, VOICE_MAP_PATH, RESPONSE_FOLDER
from .utils import get_file_path

class TTSService:
    """
    Service for synthesizing speech audio from text.
    """
    def __init__(self):
        """
        Initialize TTSService with an empty in-memory voice map.
        """
        self.voice_map: Dict[str, str] = {}

    def load(self):
        """
        Load a mapping of speaker IDs to TTS voice IDs from file.
        """
        if os.path.exists(VOICE_MAP_PATH):
            try:
                with open(VOICE_MAP_PATH, "r", encoding="utf-8") as f:
                    self.voice_map = json.load(f)
            except Exception:
                self.voice_map = {}

    def synthesize(self, text: str, base_filename: str, speaker_id: str) -> str:
        """
        Generate a TTS audio file from the given text.

        Parameters
        ----------
        text : str
            The text to convert to speech.
        base_filename : str
            The base name derived from the original audio or request.
        speaker_id : str
            The unique identifier of the speaker for voice mapping.

        Returns
        -------
        str
            The relative URL path to the generated audio file.
        """
        unique = f"{int(time.time()*1000)}_{uuid.uuid4().hex[:8]}"
        out_path = get_file_path(
            f"{base_filename}_{speaker_id}_{unique}.mp3",
            RESPONSE_FOLDER
        )
        # Default to gTTS synthesis for simplicity
        gTTS(text=text, lang="en").save(out_path)
        return f"/audio/{os.path.basename(out_path)}"
