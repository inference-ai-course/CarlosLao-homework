"""
Voice Assistant Server (FastAPI)

Features:
- ASR: Whisper
- Speaker identification: Resemblyzer + cosine similarity
- LLM: Hugging Face Transformers (text-generation pipeline)
- TTS: ElevenLabs (optional) with gTTS fallback
- Conversation memory with windowing and transcript export
- Health endpoints and static UI mount

Notes:
- Requires FFmpeg installed and available in PATH.
- HUGGINGFACE_TOKEN is required for the specified model.
- ELEVENLABS_API_KEY is optional (otherwise gTTS is used).
"""

import os
import io
import json
import logging
import tempfile
import subprocess
import wave
import pickle
import asyncio
import uuid
import time
import mimetypes
from contextlib import asynccontextmanager, closing
from typing import List, Dict, Optional

import numpy as np
import requests
import whisper
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline
from gtts import gTTS
from resemblyzer import VoiceEncoder, preprocess_wav
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Configuration and constants
# -----------------------------

load_dotenv()
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
TRANSCRIPT_FILE = os.getenv("TRANSCRIPT_FILE", "transcript.json")
RESPONSE_FOLDER = os.getenv("RESPONSE_FOLDER", "response")
SPEAKER_DB_PATH = "speakers.pkl"
VOICE_MAP_PATH = "voice_map.json"
WEB_DIR = "web"

# Windowed memory: MAX_TURNS means user+assistant pairs; total stored = 2 * MAX_TURNS
MAX_TURNS = 5

# Accept list is intentionally broad; gate with simple checks
ACCEPTED_AUDIO_TYPES = {
    "audio/mpeg",
    "audio/wav",
    "audio/x-wav",
    "audio/x-m4a",
    "audio/ogg",
    "audio/webm",
    "application/octet-stream",
}

DEFAULT_SYSTEM_PROMPT = (
    "You are a concise, helpful voice assistant. Use the conversation history to stay in context. "
    "If information is missing, ask a brief clarifying question."
)

if not HUGGINGFACE_TOKEN:
    raise RuntimeError("Missing Hugging Face token (HUGGINGFACE_TOKEN)")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("voice-assistant")

# -----------------------------
# Globals initialized at startup
# -----------------------------

asr_model = None
llm = None
encoder = VoiceEncoder()

# Conversation memory: list of {"role": "user"|"assistant"|"system", "text": str, "speaker": Optional[str]}
conversation_history: List[Dict[str, str]] = []

# Locks
history_lock = asyncio.Lock()
speakers_lock = asyncio.Lock()

# Cached voice map (speaker_id -> elevenlabs_voice_id or "default")
VOICE_MAP: Dict[str, str] = {}

# -----------------------------
# Utility functions (self-contained)
# -----------------------------

def ensure_dir(path: str) -> str:
    """Ensure directory exists; return absolute path."""
    abs_path = os.path.abspath(path)
    os.makedirs(abs_path, exist_ok=True)
    return abs_path

def get_script_dir(relative: Optional[str] = None) -> str:
    """Return absolute path for a directory relative to the script."""
    base = os.path.dirname(os.path.abspath(__file__))
    if relative:
        full = os.path.join(base, relative)
        ensure_dir(full)
        return full
    return base

def get_file_path(filename: str, base_dir: Optional[str] = None) -> str:
    """Return absolute path for a file in base_dir (created if needed)."""
    if base_dir:
        dir_abs = get_script_dir(base_dir)
        return os.path.join(dir_abs, filename)
    return os.path.join(get_script_dir(), filename)

def guess_media_type(path: str) -> str:
    """Guess media type by extension; default to audio/mpeg for .mp3, audio/wav for .wav."""
    ctype, _ = mimetypes.guess_type(path)
    if ctype:
        return ctype
    # Fallbacks
    if path.lower().endswith(".mp3"):
        return "audio/mpeg"
    if path.lower().endswith(".wav"):
        return "audio/wav"
    return "application/octet-stream"

# -----------------------------
# Speaker database and embeddings
# -----------------------------

def get_embedding(audio_path: str) -> np.ndarray:
    """Compute voice embedding for speaker identification."""
    wav = preprocess_wav(audio_path)
    return encoder.embed_utterance(wav)

def load_speakers() -> Dict[str, np.ndarray]:
    """Load speaker DB from pickle; returns dict of {speaker_id: embedding}."""
    if os.path.exists(SPEAKER_DB_PATH):
        with open(SPEAKER_DB_PATH, "rb") as f:
            return pickle.load(f)
    return {}

def save_speakers(speakers: Dict[str, np.ndarray]) -> None:
    """Persist speaker DB atomically."""
    tmp_path = f"{SPEAKER_DB_PATH}.tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(speakers, f)
    os.replace(tmp_path, SPEAKER_DB_PATH)

async def match_or_create_speaker_async(embedding: np.ndarray, threshold: float = 0.75) -> str:
    """
    Return existing speaker_id if cosine similarity > threshold; otherwise create a new one.
    Protected by a lock to prevent concurrent DB corruption.
    """
    async with speakers_lock:
        known = load_speakers()
        for speaker_id, known_embedding in known.items():
            sim = float(cosine_similarity([embedding], [known_embedding])[0][0])
            if sim > threshold:
                return speaker_id
        new_id = f"user_{len(known) + 1}"
        known[new_id] = embedding
        save_speakers(known)
        return new_id

# -----------------------------
# Voice map
# -----------------------------

def load_voice_map() -> Dict[str, str]:
    """Load a mapping from speaker_id to voice_id (for ElevenLabs)."""
    if os.path.exists(VOICE_MAP_PATH):
        try:
            with open(VOICE_MAP_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except Exception:
            logger.exception("Failed to load voice map; defaulting to empty")
    return {}

def get_voice_for_speaker(speaker_id: str) -> str:
    """Return configured voice_id for a speaker; 'default' if not found."""
    # Use cached VOICE_MAP
    return VOICE_MAP.get(speaker_id, "default")

# -----------------------------
# Prompting and memory
# -----------------------------

def normalize_role(role: str) -> str:
    """Map arbitrary role strings to canonical: User, Assistant, System."""
    role = (role or "").lower().strip()
    mapping = {"user": "User", "assistant": "Assistant", "system": "System"}
    return mapping.get(role, "User")

def get_recent_history() -> List[Dict[str, str]]:
    """Return the windowed recent history (2 * MAX_TURNS entries)."""
    return conversation_history[-(2 * MAX_TURNS):]

def format_prompt(system_message: str = DEFAULT_SYSTEM_PROMPT, include_speaker: bool = True) -> str:
    """
    Format the prompt from recent conversation history.
    Optionally includes speaker tags for user turns as User(user_1): ...
    """
    lines = [f"System: {system_message}", ""]
    for m in get_recent_history():
        role = normalize_role(m.get("role"))
        text = (m.get("text") or "").strip()
        speaker = m.get("speaker")
        prefix = f"{role}({speaker})" if include_speaker and role == "User" and speaker else role
        lines.append(f"{prefix}: {text}")
    lines.append("Assistant:")
    return "\n".join(lines)

# -----------------------------
# Readiness checks
# -----------------------------

def check_ffmpeg_installed() -> bool:
    """Check ffmpeg availability."""
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except Exception as e:
        logger.error("FFmpeg check failed: %s", e)
        return False

def check_asr_model_loaded() -> bool:
    """Try a tiny dummy transcription to verify Whisper model is responsive."""
    path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            path = tmp.name
            with closing(wave.open(path, "w")) as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(b"\x00\x00" * 16000)
        asr_model.transcribe(path, verbose=False)
        return True
    except Exception as e:
        logger.error("ASR readiness check failed: %s", e)
        return False
    finally:
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except Exception:
                pass

def check_llm_ready() -> bool:
    """Ping the LLM for a tiny generation."""
    try:
        result = llm("Hello", max_new_tokens=5, return_full_text=False)
        # HF pipeline returns a list of dicts with 'generated_text'
        if result and isinstance(result, list) and result[0].get("generated_text"):
            return True
        return False
    except Exception as e:
        logger.error("LLM readiness check failed: %s", e)
        return False

def check_gtts_ready() -> bool:
    """Attempt a minimal gTTS synthesis to a temp file."""
    path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            path = tmp.name
        gTTS("OK", lang="en").save(path)
        return os.path.exists(path) and os.path.getsize(path) > 0
    except Exception as e:
        logger.error("gTTS readiness check failed: %s", e)
        return False
    finally:
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except Exception:
                pass

def ensure_ready() -> None:
    """Raise HTTPException if any critical dependency is missing."""
    if not check_ffmpeg_installed():
        logger.error("Readiness failed: FFmpeg not found")
        raise HTTPException(500, "FFmpeg not found")
    if not check_asr_model_loaded():
        logger.error("Readiness failed: ASR check failed")
        raise HTTPException(500, "ASR check failed")
    if not check_llm_ready():
        logger.error("Readiness failed: LLM not generating")
        raise HTTPException(500, "LLM not generating")
    if not check_gtts_ready():
        logger.error("Readiness failed: gTTS failed")
        raise HTTPException(500, "gTTS failed")

# -----------------------------
# Core pipeline steps
# -----------------------------

async def save_temp_audio(file: UploadFile) -> str:
    """Save uploaded audio to a temporary file with the same extension if present."""
    suffix = os.path.splitext((file.filename or "").strip())[1] or ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        data = await file.read()
        tmp.write(data)
        path = tmp.name
    logger.info("Saved temporary audio: %s", path)
    return path

def transcribe_audio_sync(audio_path: str) -> str:
    """Transcribe audio and delete the temp file afterwards."""
    try:
        result = asr_model.transcribe(audio_path, verbose=False)
        text = (result.get("text") or "").strip()
        logger.info("Transcription length: %d", len(text))
        return text
    except Exception as e:
        logger.exception("Transcription failed")
        raise RuntimeError(f"Transcription failed: {e}")
    finally:
        try:
            os.remove(audio_path)
        except Exception:
            pass

def llm_generate_sync(prompt: str) -> str:
    """
    Generate assistant reply from prompt.
    The pipeline is configured to return only the continuation.
    """
    outputs = llm(prompt, max_new_tokens=200, do_sample=True, temperature=0.7, return_full_text=False)
    text = (outputs[0].get("generated_text") or "").strip()
    # Remove a leading "Assistant:" if the model includes it
    if text.lower().startswith("assistant:"):
        text = text[len("assistant:"):].strip()
    logger.info("Generated reply length: %d", len(text))
    return text

def synthesize_speech_sync(text: str, input_filename: str, speaker_id: str) -> str:
    """
    Synthesize speech for assistant's reply.
    Returns a URL path (e.g., /audio/filename.mp3) to fetch the audio.
    """
    base_name = os.path.splitext(os.path.basename(input_filename or ""))[0] or "response"
    unique = f"{int(time.time()*1000)}_{uuid.uuid4().hex[:8]}"
    out_path = get_file_path(f"{base_name}_{speaker_id}_{unique}.mp3", RESPONSE_FOLDER)

    voice_id = get_voice_for_speaker(speaker_id)
    if voice_id == "default" or not ELEVENLABS_API_KEY:
        # Fallback to gTTS
        gTTS(text=text, lang="en").save(out_path)
        logger.info("Synthesized TTS via gTTS: %s", out_path)
        return f"/audio/{os.path.basename(out_path)}"

    # Try ElevenLabs first
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {"xi-api-key": ELEVENLABS_API_KEY}
    payload = {"text": text}
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        if not response.ok or not response.content:
            logger.warning("ElevenLabs TTS failed with status %s; falling back to gTTS", response.status_code)
            gTTS(text=text, lang="en").save(out_path)
        else:
            with open(out_path, "wb") as f:
                f.write(response.content)
            logger.info("Synthesized TTS via ElevenLabs: %s", out_path)
    except Exception:
        logger.exception("ElevenLabs TTS exception; falling back to gTTS")
        gTTS(text=text, lang="en").save(out_path)

    return f"/audio/{os.path.basename(out_path)}"

def save_chat_transcript_sync() -> str:
    """Persist the recent history window to TRANSCRIPT_FILE."""
    transcript_path = get_file_path(TRANSCRIPT_FILE)
    with open(transcript_path, "w", encoding="utf-8") as f:
        json.dump(get_recent_history(), f, indent=2, ensure_ascii=False)
    logger.info("Saved transcript: %s", transcript_path)
    return transcript_path

# -----------------------------
# FastAPI app and lifespan
# -----------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup:
    - Load Whisper model
    - Initialize HF text-generation pipeline
    - Ensure output directories
    - Prime VOICE_MAP
    Shutdown: nothing special
    """
    global asr_model, llm, VOICE_MAP

    # ASR model (choose size based on latency needs; "small" is a reasonable default)
    asr_model = whisper.load_model("small")

    # LLM pipeline
    # Tip: if you have accelerate installed and a GPU, consider device_map="auto".
    # We guard in try/except to avoid import/attr issues if accelerate is not present.
    llm_kwargs = dict(model="meta-llama/Llama-3.1-8B-Instruct", token=HUGGINGFACE_TOKEN)
    try:
        # Optional: device_map and torch_dtype can improve performance if available
        llm = pipeline("text-generation", return_full_text=False, **llm_kwargs)  # type: ignore
    except Exception:
        # Fallback without extra kwargs
        llm = pipeline("text-generation", **llm_kwargs)  # type: ignore

    # Ensure folders exist
    ensure_dir(get_file_path("", RESPONSE_FOLDER))
    ensure_dir(get_file_path("", WEB_DIR))

    # Load voice map once
    VOICE_MAP = load_voice_map()

    yield

app = FastAPI(title="Voice Assistant", version="3.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Static UI (optional). Place your frontend in ./web
web_path = get_file_path("", WEB_DIR)
app.mount("/ui", StaticFiles(directory=web_path, html=True), name="web")

# -----------------------------
# Schemas
# -----------------------------

class ChatResponse(BaseModel):
    speaker_id: str
    transcription: str
    response: str
    tts_audio: str
    conversation_memory: List[Dict[str, str]]
    debug_prompt: str

class HealthResponse(BaseModel):
    ffmpeg_installed: bool
    asr_model_loaded: bool
    llm_ready: bool
    gtts_ready: bool

# -----------------------------
# Routes
# -----------------------------

@app.post("/chat/", response_model=ChatResponse)
async def chat(file: UploadFile = File(...)):
    """
    Upload an audio file and receive:
    - Speaker ID
    - Transcribed text
    - Assistant text response
    - TTS audio URL
    - Recent conversation memory
    - Prompt used for generation (for debugging)
    """
    if file.content_type not in ACCEPTED_AUDIO_TYPES:
        raise HTTPException(415, f"Unsupported media type: {file.content_type}")

    ensure_ready()

    # Save input audio
    audio_path = await save_temp_audio(file)

    # Speaker embedding and ID
    embedding = await asyncio.to_thread(get_embedding, audio_path)
    speaker_id = await match_or_create_speaker_async(embedding)
    logger.info("Speaker identified: %s", speaker_id)

    # ASR
    user_text = await asyncio.to_thread(transcribe_audio_sync, audio_path)

    # Update memory and build prompt
    async with history_lock:
        conversation_history.append({"role": "user", "text": user_text, "speaker": speaker_id})
        if len(conversation_history) > MAX_TURNS * 2:
            del conversation_history[: len(conversation_history) - MAX_TURNS * 2]
        prompt = format_prompt()

    # LLM then TTS on the assistant reply
    assistant_reply = await asyncio.to_thread(llm_generate_sync, prompt)
    tts_path = await asyncio.to_thread(synthesize_speech_sync, assistant_reply, file.filename or "", speaker_id)

    # Persist memory and snapshot
    async with history_lock:
        conversation_history.append({"role": "assistant", "text": assistant_reply})
        memory_snapshot = list(get_recent_history())

    await asyncio.to_thread(save_chat_transcript_sync)

    return JSONResponse(
        content={
            "speaker_id": speaker_id,
            "transcription": user_text,
            "response": assistant_reply,
            "tts_audio": tts_path,
            "conversation_memory": memory_snapshot,
            "debug_prompt": prompt,
        }
    )

@app.get("/audio/{filename}")
async def get_audio(filename: str):
    """
    Serve synthesized audio by filename.
    """
    file_path = get_file_path(filename, RESPONSE_FOLDER)
    if not os.path.exists(file_path):
        raise HTTPException(404, "Audio file not found")
    return FileResponse(file_path, media_type=guess_media_type(file_path))

@app.get("/memory")
async def memory():
    """
    Return the recent conversation memory window.
    """
    async with history_lock:
        return get_recent_history()

@app.delete("/memory")
async def clear_memory():
    """
    Clear in-memory history and delete the transcript file if present.
    """
    async with history_lock:
        conversation_history.clear()
    try:
        path_resolved = get_file_path(TRANSCRIPT_FILE)
        if os.path.exists(path_resolved):
            os.remove(path_resolved)
        elif os.path.exists(TRANSCRIPT_FILE):
            os.remove(TRANSCRIPT_FILE)
    except Exception:
        pass
    return {"cleared": True}

@app.get("/health", response_model=HealthResponse)
def health():
    """
    Lightweight liveness/health. Does not perform heavy operations.
    """
    return {
        "ffmpeg_installed": check_ffmpeg_installed(),
        "asr_model_loaded": asr_model is not None,
        "llm_ready": llm is not None,
        "gtts_ready": True,  # gTTS is local and will be exercised in /health/ready
    }

@app.get("/health/ready", response_model=HealthResponse)
def health_ready():
    """
    Heavier readiness checks. Calls into ASR, LLM, and TTS code paths.
    """
    return {
        "ffmpeg_installed": check_ffmpeg_installed(),
        "asr_model_loaded": check_asr_model_loaded(),
        "llm_ready": check_llm_ready(),
        "gtts_ready": check_gtts_ready(),
    }

# -----------------------------
# Entrypoint
# -----------------------------

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    logger.info("Starting server on 0.0.0.0:%d", port)
    uvicorn.run(app, host="0.0.0.0", port=port)
