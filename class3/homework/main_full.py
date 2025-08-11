"""
Voice Assistant API

This FastAPI service turns short audio clips into a conversational exchange:
- Speech-to-text (ASR) using OpenAI Whisper (local model).
- Response generation using a Hugging Face text-generation pipeline (Llama Instruct).
- Text-to-speech (TTS) using gTTS to produce an audio reply.
- Lightweight conversation memory, transcript persistence, and health checks.

Endpoints
- POST /chat/: Accepts an audio file upload, returns transcription, generated reply, and TTS file path.
- GET  /health: Reports readiness for FFmpeg, ASR, LLM, and gTTS.

Environment Variables
- HUGGINGFACE_TOKEN (required): Authentication token for loading/generating with the HF model.
- TRANSCRIPT_FILE (optional): Filename for saving the rolling conversation transcript (default: "transcript.txt").
- RESPONSE_FOLDER (optional): Destination folder for synthesized TTS files (default: "response").
- PORT (optional): Port for uvicorn to bind to (default: 8000).

Dependencies and External Requirements
- FFmpeg must be installed and in PATH for audio handling.
- Models:
  - Whisper small model (downloaded by whisper.load_model).
  - Hugging Face text-generation model: meta-llama/Llama-3.1-8B-Instruct
- gTTS requires network access to generate audio.
- A local `utils` module is expected to provide:
  - get_script_dir(subdir: str): ensures a subdirectory exists relative to the script.
  - get_file_path(filename: str, subdir: Optional[str] = None): returns a file path rooted relative to script dir.

Notes
- This service maintains a short rolling conversation history (MAX_TURNS), enabling brief multi-turn context.
- Health checks perform live, minimal end-to-end tests for each component.
"""

import json
import logging
import os
import subprocess
import tempfile
import wave
from contextlib import asynccontextmanager, closing
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from gtts import gTTS
from transformers import pipeline
import whisper

import utils

# -----------------------------------------------------------------------------
# Configuration and logging
# -----------------------------------------------------------------------------

load_dotenv()
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
TRANSCRIPT_FILE = os.getenv("TRANSCRIPT_FILE", "transcript.txt")  # JSON content may be written with this extension
RESPONSE_FOLDER = os.getenv("RESPONSE_FOLDER", "response")

if not HUGGINGFACE_TOKEN:
    raise RuntimeError("Missing Hugging Face token. Please set HUGGINGFACE_TOKEN.")

# Configure logging early. INFO-level is appropriate for service logs.
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("voice-assistant")

# -----------------------------------------------------------------------------
# Global state
# -----------------------------------------------------------------------------

asr_model: Optional[Any] = None  # Whisper model instance
llm: Optional[Any] = None        # HF text-generation pipeline
conversation_history: List[Dict[str, str]] = []
MAX_TURNS: int = 5               # Number of conversational turns to retain (user+assistant pairs)

# Accepted MIME types for uploaded audio files
ACCEPTED_AUDIO_TYPES = {
    "audio/mpeg",
    "audio/wav",
    "audio/x-wav",
    "audio/x-m4a",
    "audio/ogg",
    "audio/webm",
    "application/octet-stream",
}

# -----------------------------------------------------------------------------
# Application lifecycle
# -----------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context:
    - Load the ASR and LLM models once at startup.
    - Ensure the response directory exists.
    """
    global asr_model, llm
    logger.info("Starting application lifespan initialization.")
    # Load Whisper ASR model (local CPU/GPU depending on availability)
    asr_model = whisper.load_model("small")
    logger.info("Whisper ASR model loaded: small")

    # Initialize Hugging Face text-generation pipeline
    llm = pipeline(
        "text-generation",
        model="meta-llama/Llama-3.1-8B-Instruct",
        token=HUGGINGFACE_TOKEN,
    )
    logger.info("Hugging Face text-generation pipeline initialized: meta-llama/Llama-3.1-8B-Instruct")

    # Ensure the response folder exists relative to script directory
    utils.get_script_dir(RESPONSE_FOLDER)
    logger.info("Response directory ensured: %s", RESPONSE_FOLDER)

    yield

    logger.info("Application lifespan shutdown complete.")


app = FastAPI(title="Voice Assistant", version="2.0", lifespan=lifespan)

# -----------------------------------------------------------------------------
# Readiness checks
# -----------------------------------------------------------------------------

def check_ffmpeg_installed() -> bool:
    """
    Verify that FFmpeg is available on PATH.
    Returns True if found, False otherwise.
    """
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except Exception as e:
        logger.error("FFmpeg check failed: %s", e)
        return False


def check_asr_model_loaded() -> bool:
    """
    Verify Whisper ASR model can process a trivial audio input.
    Creates a short silent WAV, transcribes, and cleans up.
    """
    path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            path = tmp.name
            with closing(wave.open(path, "w")) as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                # 1 second of silence at 16kHz, 16-bit mono
                wf.writeframes(b"\x00\x00" * 16000)
        asr_model.transcribe(path, verbose=False)  # type: ignore[union-attr]
        return True
    except Exception as e:
        logger.error("ASR readiness check failed: %s", e)
        return False
    finally:
        if path:
            try:
                os.remove(path)
            except Exception:
                # Non-fatal cleanup failure
                pass


def check_llm_ready() -> bool:
    """
    Verify text-generation pipeline is responsive and returns output.
    """
    try:
        result = llm("Hello", max_new_tokens=5)  # type: ignore[operator]
        return bool(result and result[0].get("generated_text"))
    except Exception as e:
        logger.error("LLM readiness check failed: %s", e)
        return False


def check_gtts_ready() -> bool:
    """
    Verify gTTS can synthesize a trivial utterance and save to disk.
    """
    path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            path = tmp.name
        gTTS("OK", lang="en").save(path)
        return os.path.exists(path) and os.path.getsize(path) > 0
    except Exception as e:
        logger.error("gTTS readiness check failed: %s", e)
        return False
    finally:
        if path:
            try:
                os.remove(path)
            except Exception:
                pass


def ensure_ready() -> None:
    """
    Perform all readiness checks and raise HTTPException if any fail.
    """
    if not check_ffmpeg_installed():
        raise HTTPException(500, "FFmpeg not found")
    if not check_asr_model_loaded():
        raise HTTPException(500, "ASR check failed")
    if not check_llm_ready():
        raise HTTPException(500, "LLM not generating")
    if not check_gtts_ready():
        raise HTTPException(500, "gTTS failed")

# -----------------------------------------------------------------------------
# Core utilities
# -----------------------------------------------------------------------------

async def save_temp_audio(file: UploadFile) -> str:
    """
    Persist uploaded audio to a temporary file and return the file path.

    Validates content type and extension to allow common audio formats.
    """
    if file.content_type not in ACCEPTED_AUDIO_TYPES and not file.filename.lower().endswith(
        (".mp3", ".wav", ".m4a", ".ogg", ".webm")
    ):
        logger.warning("Unsupported file format: %s (%s)", file.filename, file.content_type)
        raise HTTPException(400, "Unsupported file format")

    suffix = os.path.splitext(file.filename)[1] or ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        data = await file.read()
        tmp.write(data)
        logger.info("Saved uploaded audio to %s (%d bytes)", tmp.name, len(data))
        return tmp.name


def transcribe_audio(audio_path: str) -> str:
    """
    Transcribe the audio file at the given path using Whisper.

    Always attempts to remove the temporary audio file after processing.
    """
    logger.info("Transcribing %s", audio_path)
    try:
        result = asr_model.transcribe(audio_path, verbose=False)  # type: ignore[union-attr]
        text = result.get("text", "").strip()
        logger.info("Transcription complete. Characters: %d", len(text))
        return text
    except Exception as e:
        logger.error("Transcription failed: %s", e)
        raise RuntimeError(f"Transcription failed: {e}")
    finally:
        try:
            os.remove(audio_path)
            logger.info("Removed temporary audio file: %s", audio_path)
        except Exception:
            # Non-fatal cleanup failure
            pass


def generate_response(user_prompt: str) -> str:
    """
    Generate an assistant reply conditioned on the rolling conversation history.

    - Appends the user's prompt to history.
    - Trims history to the last MAX_TURNS pairs.
    - Builds a conversational prompt and queries the LLM.
    - Extracts the assistant's next line of text.
    """
    conversation_history.append({"role": "user", "text": user_prompt})
    if len(conversation_history) > MAX_TURNS * 2:
        conversation_history[:] = conversation_history[-MAX_TURNS * 2 :]

    prompt = "\n".join(f"{t['role']}: {t['text']}" for t in conversation_history) + "\nassistant:"
    logger.info("Generating response. Prompt length: %d", len(prompt))
    logger.debug("Prompt preview (first 200 chars): %s", prompt[:200])

    outputs = llm(prompt, max_new_tokens=150, do_sample=True, temperature=0.7)  # type: ignore[operator]
    full_text = outputs[0]["generated_text"]

    # Extract only the assistant's first line after the prompt
    reply = full_text[len(prompt):].strip().split("\n")[0]
    conversation_history.append({"role": "assistant", "text": reply})
    logger.info("Generated reply. Characters: %d", len(reply))
    logger.debug("Reply content: %s", reply)
    return reply


def save_chat_transcript() -> str:
    """
    Save the recent conversation history (rolling window) as JSON.

    Uses TRANSCRIPT_FILE to determine the output filename.
    """
    path = utils.get_file_path(TRANSCRIPT_FILE)
    recent_turns = conversation_history[-MAX_TURNS * 2 :]
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(recent_turns, f, indent=2, ensure_ascii=False)
        logger.info("Saved transcript as JSON with last %d turns to %s", MAX_TURNS, path)
    except Exception as e:
        logger.error("Failed to save JSON transcript: %s", e)
    return path


def synthesize_speech(text: str, input_filename: str) -> str:
    """
    Convert assistant text to speech using gTTS and save to the response folder.

    Output filename is derived from the original input file's base name.
    """
    base_name = os.path.splitext(os.path.basename(input_filename))[0] or "response"
    out_path = utils.get_file_path(f"{base_name}_response.mp3", RESPONSE_FOLDER)
    try:
        logger.info("Synthesizing TTS to %s", out_path)
        gTTS(text=text, lang="en").save(out_path)
        return out_path
    except Exception as e:
        logger.error("TTS failed: %s", e)
        raise RuntimeError("TTS failed")

# -----------------------------------------------------------------------------
# API endpoints
# -----------------------------------------------------------------------------

@app.post("/chat/")
async def chat(file: UploadFile = File(...)):
    """
    Process an audio file:
    1) Save temporary audio.
    2) Transcribe speech-to-text.
    3) Generate an assistant response.
    4) Save rolling transcript.
    5) Synthesize a spoken reply and return paths and text.

    Returns JSON with:
    - transcription: The user's transcribed text.
    - response: The assistant's generated reply.
    - tts_audio: Path to the synthesized reply audio file.
    """
    logger.info("Received /chat request.")
    try:
        ensure_ready()
        audio_path = await save_temp_audio(file)
        user_text = transcribe_audio(audio_path)
        assistant_reply = generate_response(user_text)
        save_chat_transcript()
        tts_path = synthesize_speech(assistant_reply, file.filename)
        logger.info("Completed /chat request.")
        return JSONResponse(
            content={
                "transcription": user_text,
                "response": assistant_reply,
                "tts_audio": tts_path,
            }
        )
    except HTTPException:
        # Already an intentional HTTP error with status code
        logger.warning("Request failed with HTTPException.")
        raise
    except Exception as e:
        logger.exception("Unhandled error processing /chat: %s", e)
        raise HTTPException(500, "Internal server error")


@app.get("/health")
def health():
    """
    Health endpoint that performs live checks against FFmpeg, ASR, LLM, and gTTS.
    """
    logger.info("Health check requested.")
    return {
        "ffmpeg_installed": check_ffmpeg_installed(),
        "asr_model_loaded": check_asr_model_loaded(),
        "llm_ready": check_llm_ready(),
        "gtts_ready": check_gtts_ready(),
    }

# -----------------------------------------------------------------------------
# Local run entrypoint
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    logger.info("Starting server on 0.0.0.0:%d", port)
    uvicorn.run(app, host="0.0.0.0", port=port)
