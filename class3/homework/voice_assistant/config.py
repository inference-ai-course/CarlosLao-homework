"""
config.py
---------
Centralized configuration constants and environment variable loading.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# API keys and tokens
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

# Files and directories
TRANSCRIPT_FILE = os.getenv("TRANSCRIPT_FILE", "transcript.json")
RESPONSE_FOLDER = os.getenv("RESPONSE_FOLDER", "response")
WEB_DIR = os.getenv("WEB_DIR", "web")
VOICE_MAP_PATH = os.getenv("VOICE_MAP_PATH", "voice_map.json")

# Models
ASR_MODEL_NAME = os.getenv("ASR_MODEL_NAME", "small")
LLM_MODEL_ID = os.getenv("LLM_MODEL_ID", "meta-llama/Llama-3.1-8B-Instruct")

# Conversation memory settings
MAX_TURNS = int(os.getenv("MAX_TURNS", "5"))

# Supported audio types
ACCEPTED_AUDIO_TYPES = {
    "audio/mpeg",        # .mp3
    "audio/wav",         # .wav
    "audio/x-wav",       # alt MIME for .wav
    "audio/ogg",         # .ogg
    "audio/x-m4a",       # .m4a
    "audio/webm",        # .webm from MediaRecorder
    "audio/flac",        # .flac
    "audio/aac",         # .aac
    "application/octet-stream",  # fallback
}

# Default persona/system prompt
DEFAULT_SYSTEM_PROMPT = (
    "You are a concise, helpful voice assistant. Use the conversation "
    "history to stay in context. If information is missing, ask a brief "
    "clarifying question."
)
