import os
import json
import logging
import tempfile
import subprocess
import wave
import pickle
import asyncio
from contextlib import asynccontextmanager, closing
from typing import List, Dict

import numpy as np
import requests
import whisper
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline
from gtts import gTTS
from resemblyzer import VoiceEncoder, preprocess_wav
from sklearn.metrics.pairwise import cosine_similarity

from utils import get_script_dir, get_file_path

load_dotenv()
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
TRANSCRIPT_FILE = os.getenv("TRANSCRIPT_FILE", "transcript.json")
RESPONSE_FOLDER = os.getenv("RESPONSE_FOLDER", "response")
SPEAKER_DB_PATH = "speakers.pkl"
VOICE_MAP_PATH = "voice_map.json"
MAX_TURNS = 5
ACCEPTED_AUDIO_TYPES = {
    "audio/mpeg",
    "audio/wav",
    "audio/x-wav",
    "audio/x-m4a",
    "audio/ogg",
    "audio/webm",
    "application/octet-stream",
}
DEFAULT_SYSTEM_PROMPT = "You are a concise, helpful voice assistant. Use the conversation history to stay in context. If information is missing, ask a brief clarifying question."

if not HUGGINGFACE_TOKEN:
    raise RuntimeError("Missing Hugging Face token")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("voice-assistant")

asr_model = None
llm = None
encoder = VoiceEncoder()
conversation_history: List[Dict[str, str]] = []
history_lock = asyncio.Lock()

def get_embedding(audio_path: str) -> np.ndarray:
    wav = preprocess_wav(audio_path)
    return encoder.embed_utterance(wav)

def load_speakers() -> Dict:
    if os.path.exists(SPEAKER_DB_PATH):
        with open(SPEAKER_DB_PATH, "rb") as f:
            return pickle.load(f)
    return {}

def save_speakers(speakers: Dict) -> None:
    with open(SPEAKER_DB_PATH, "wb") as f:
        pickle.dump(speakers, f)

def match_or_create_speaker(embedding: np.ndarray, known_speakers: Dict, threshold: float = 0.75) -> str:
    for speaker_id, known_embedding in known_speakers.items():
        if cosine_similarity([embedding], [known_embedding])[0][0] > threshold:
            return speaker_id
    new_id = f"user_{len(known_speakers)+1}"
    known_speakers[new_id] = embedding
    save_speakers(known_speakers)
    return new_id

def load_voice_map() -> Dict:
    if os.path.exists(VOICE_MAP_PATH):
        with open(VOICE_MAP_PATH, "r") as f:
            return json.load(f)
    return {}

def get_voice_for_speaker(speaker_id: str) -> str:
    return load_voice_map().get(speaker_id, "default")

def normalize_role(role: str) -> str:
    role = role.lower().strip()
    return {
        "user": "User",
        "human": "User",
        "assistant": "Assistant",
        "bot": "Assistant",
        "ai": "Assistant",
        "system": "System",
    }.get(role, role.capitalize() or "User")

def get_recent_history() -> List[Dict[str, str]]:
    return conversation_history[-(2 * MAX_TURNS):]

def format_prompt(user_text: str, system_message: str = DEFAULT_SYSTEM_PROMPT) -> str:
    lines = [f"System: {system_message}", ""]
    lines += [f"{normalize_role(m['role'])}: {m['text'].strip()}" for m in get_recent_history()]
    lines.append(f"User: {user_text.strip()}")
    lines.append("Assistant:")
    return "\n".join(lines)

def check_ffmpeg_installed() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except Exception as e:
        logger.error("FFmpeg check failed: %s", e)
        return False

def check_asr_model_loaded() -> bool:
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
    try:
        result = llm("Hello", max_new_tokens=5)
        return bool(result and result[0].get("generated_text"))
    except Exception as e:
        logger.error("LLM readiness check failed: %s", e)
        return False

def check_gtts_ready() -> bool:
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

async def save_temp_audio(file: UploadFile) -> str:
    suffix = os.path.splitext((file.filename or "").strip())[1] or ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        data = await file.read()
        tmp.write(data)
        path = tmp.name
    logger.info("Saved temporary audio: %s", path)
    return path

def transcribe_audio_sync(audio_path: str) -> str:
    try:
        result = asr_model.transcribe(audio_path, verbose=False)
        text = result.get("text", "").strip()
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
    outputs = llm(prompt, max_new_tokens=150, do_sample=True, temperature=0.7)
    full_text = outputs[0].get("generated_text", "")
    tail = full_text[len(prompt):].strip() if full_text.startswith(prompt) else full_text.strip()
    for line in tail.splitlines():
        line = line.strip()
        if line:
            logger.info("Generated reply length: %d", len(line))
            return line
    logger.info("Generated reply length: %d", len(tail))
    return tail

def synthesize_speech_sync(text: str, input_filename: str, speaker_id: str) -> str:
    base_name = os.path.splitext(os.path.basename(input_filename or ""))[0] or RESPONSE_FOLDER
    out_path = get_file_path(f"{base_name}_{speaker_id}_response.mp3", RESPONSE_FOLDER)
    voice_id = get_voice_for_speaker(speaker_id)
    if voice_id == "default" or not ELEVENLABS_API_KEY:
        gTTS(text=text, lang="en").save(out_path)
        logger.info("Synthesized TTS via gTTS: %s", out_path)
        return f"/audio/{os.path.basename(out_path)}"
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
    transcript_path = get_file_path(TRANSCRIPT_FILE)
    with open(transcript_path, "w", encoding="utf-8") as f:
        json.dump(get_recent_history(), f, indent=2, ensure_ascii=False)
    logger.info("Saved transcript: %s", transcript_path)
    return transcript_path

@asynccontextmanager
async def lifespan(app: FastAPI):
    global asr_model, llm
    asr_model = whisper.load_model("small")
    llm = pipeline("text-generation", model="meta-llama/Llama-3.1-8B-Instruct", token=HUGGINGFACE_TOKEN)
    get_script_dir(RESPONSE_FOLDER)
    yield

app = FastAPI(title="Voice Assistant", version="3.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
web_path = get_file_path("web")
app.mount("/ui", StaticFiles(directory=web_path, html=True), name="web")

@app.post("/chat/")
async def chat(file: UploadFile = File(...)):
    ensure_ready()
    audio_path = await save_temp_audio(file)
    embedding = await asyncio.to_thread(get_embedding, audio_path)
    known_speakers = load_speakers()
    speaker_id = await asyncio.to_thread(match_or_create_speaker, embedding, known_speakers)
    logger.info("Speaker identified: %s", speaker_id)
    user_text = await asyncio.to_thread(transcribe_audio_sync, audio_path)
    async with history_lock:
        conversation_history.append({"role": speaker_id, "text": user_text})
        if len(conversation_history) > MAX_TURNS * 2:
            del conversation_history[: len(conversation_history) - MAX_TURNS * 2]
        prompt = format_prompt(user_text)
    llm_task = asyncio.to_thread(llm_generate_sync, prompt)
    tts_task = asyncio.to_thread(synthesize_speech_sync, user_text, file.filename or "", speaker_id)
    assistant_reply, tts_path = await asyncio.gather(llm_task, tts_task)
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
    file_path = get_file_path(filename, RESPONSE_FOLDER)
    if not os.path.exists(file_path):
        raise HTTPException(404, "Audio file not found")
    return FileResponse(file_path, media_type="audio/mpeg")

@app.get("/memory")
async def memory():
    async with history_lock:
        return get_recent_history()

@app.delete("/memory")
async def clear_memory():
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

@app.get("/health")
def health():
    return {
        "ffmpeg_installed": check_ffmpeg_installed(),
        "asr_model_loaded": check_asr_model_loaded(),
        "llm_ready": check_llm_ready(),
        "gtts_ready": check_gtts_ready(),
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    logger.info("Starting server on 0.0.0.0:%d", port)
    uvicorn.run(app, host="0.0.0.0", port=port)
