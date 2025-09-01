"""
asr.py
-------
Automatic Speech Recognition (ASR) module.

This module provides the ASRService class which wraps a Whisper model instance
for converting spoken audio into transcribed text. The class is designed
to be initialized once and reused to avoid repeated model loading.
"""

import whisper
from typing import Optional

class ASRService:
    """
    Service for performing speech-to-text transcription using Whisper.
    """
    def __init__(self, model_name: str):
        """
        Initialize the ASRService with a given Whisper model name.

        Parameters
        ----------
        model_name : str
            The model identifier (e.g., "base", "small", "medium", "large").
        """
        self.model_name = model_name
        self.model: Optional[any] = None

    def load(self):
        """
        Load the Whisper model into memory.
        This should be called once at application startup.
        """
        self.model = whisper.load_model(self.model_name)

    def transcribe(self, audio_path: str) -> str:
        """
        Transcribe an audio file into text.

        Parameters
        ----------
        audio_path : str
            Path to the audio file to be transcribed.

        Returns
        -------
        str
            The transcribed text.
        """
        if self.model is None:
            raise RuntimeError("ASR model not loaded")
        result = self.model.transcribe(audio_path, verbose=False)
        return (result.get("text") or "").strip()
