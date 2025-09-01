"""
health.py
---------
Health check utilities for validating external dependencies and components.
"""

import subprocess
import tempfile
import wave
import os
from contextlib import closing
from gtts import gTTS

def check_ffmpeg_installed() -> bool:
    """
    Determine if ffmpeg is installed and available on the system.
    """
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except Exception:
        return False

def check_asr_ready(asr_service) -> bool:
    """
    Validate ASR by attempting a dummy transcription.
    """
    path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            path = tmp.name
            with closing(wave.open(path, "w")) as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(b"\x00\x00" * 16000)
        _ = asr_service.transcribe(path)
        return True
    except Exception:
        return False
    finally:
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except Exception:
                pass

def check_llm_ready(llm_service) -> bool:
    """
    Validate LLM readiness by attempting a small generation.
    """
    try:
        text = llm_service.generate("Hello", max_new_tokens=5)
        return bool(text)
    except Exception:
        return False

def check_gtts_ready() -> bool:
    """
    Validate Google TTS synthesis by generating a short MP3 to a temp file.

    Returns
    -------
    bool
        True if a file is successfully created and non-empty, False otherwise.
    """
    path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            path = tmp.name
        gTTS("OK", lang="en").save(path)
        return os.path.exists(path) and os.path.getsize(path) > 0
    except Exception:
        return False
    finally:
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except Exception:
                pass

