import json
import logging
import os
import subprocess
import tempfile
import wave
import asyncio
from contextlib import asynccontextmanager, closing
from typing import List, Dict

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from transformers import pipeline
from dotenv import load_dotenv
import whisper
from gtts import gTTS
import utils

load_dotenv()
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
TRANSCRIPT_FILE = os.getenv("TRANSCRIPT_FILE", "transcript.json")
RESPONSE_FOLDER = os.getenv("RESPONSE_FOLDER", "response")

if not HUGGINGFACE_TOKEN:
    raise RuntimeError("Missing Hugging Face token. Please set HUGGINGFACE_TOKEN.")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("voice-assistant")

asr_model = None
llm = None
conversation_history: List[Dict[str, str]] = []
history_lock = asyncio.Lock()
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

DEFAULT_SYSTEM = (
    "You are a concise, helpful voice assistant. "
    "Use the conversation history to stay in context. "
    "If information is missing, ask a brief clarifying question."
)

def normalize_role(role: str) -> str:
    role = role.lower().strip()
    if role in ("user", "human"):
        return "User"
    if role in ("assistant", "bot", "ai"):
        return "Assistant"
    if role in ("system",):
        return "System"
    return role.capitalize() or "User"

def get_recent_history() -> List[Dict[str, str]]:
    return conversation_history[-(2 * MAX_TURNS):]

def format_prompt(user_text: str, system_message: str = DEFAULT_SYSTEM) -> str:
    lines = [f"System: {system_message}", ""]
    for m in get_recent_history():
        lines.append(f"{normalize_role(m['role'])}: {m['text'].strip()}")
    lines.append(f"User: {user_text.strip()}")
    lines.append("Assistant:")
    return "\n".join(lines)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global asr_model, llm
    asr_model = whisper.load_model("small")
    llm = pipeline(
        "text-generation",
        model="meta-llama/Llama-3.1-8B-Instruct",
        token=HUGGINGFACE_TOKEN
    )
    utils.get_script_dir(RESPONSE_FOLDER)
    yield

app = FastAPI(title="Voice Assistant", version="2.1", lifespan=lifespan)

def check_ffmpeg_installed():
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except:
        return False

def check_asr_model_loaded():
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
        if path:
            try:
                os.remove(path)
            except:
                pass

def check_llm_ready():
    try:
        result = llm("Hello", max_new_tokens=5)
        return bool(result and result[0].get("generated_text"))
    except Exception as e:
        logger.error("LLM readiness check failed: %s", e)
        return False

def check_gtts_ready():
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
        if path:
            try:
                os.remove(path)
            except:
                pass

def ensure_ready():
    if not check_ffmpeg_installed():
        raise HTTPException(500, "FFmpeg not found")
    if not check_asr_model_loaded():
        raise HTTPException(500, "ASR check failed")
    if not check_llm_ready():
        raise HTTPException(500, "LLM not generating")
    if not check_gtts_ready():
        raise HTTPException(500, "gTTS failed")

async def save_temp_audio(file: UploadFile) -> str:
    if file.content_type not in ACCEPTED_AUDIO_TYPES and not file.filename.lower().endswith((".mp3", ".wav", ".m4a", ".ogg", ".webm")):
        raise HTTPException(400, "Unsupported file format")
    suffix = os.path.splitext(file.filename)[1] or ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        data = await file.read()
        tmp.write(data)
        return tmp.name

def transcribe_audio_sync(audio_path: str) -> str:
    try:
        result = asr_model.transcribe(audio_path, verbose=False)
        text = result.get("text", "").strip()
        return text
    except Exception as e:
        raise RuntimeError(f"Transcription failed: {e}")
    finally:
        try:
            os.remove(audio_path)
        except:
            pass

def llm_generate_sync(prompt: str) -> str:
    outputs = llm(prompt, max_new_tokens=150, do_sample=True, temperature=0.7)
    full_text = outputs[0]["generated_text"]
    reply = full_text[len(prompt):].strip()
    reply = reply.split("\n")[0].strip()
    return reply

def synthesize_speech_sync(text: str, input_filename: str) -> str:
    base_name = os.path.splitext(os.path.basename(input_filename))[0] or "response"
    out_path = utils.get_file_path(f"{base_name}_response.mp3", RESPONSE_FOLDER)
    gTTS(text=text, lang="en").save(out_path)
    return out_path

def save_chat_transcript_sync() -> str:
    path = utils.get_file_path(TRANSCRIPT_FILE)
    recent_turns = get_recent_history()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(recent_turns, f, indent=2, ensure_ascii=False)
    return path

@app.post("/chat/")
async def chat(file: UploadFile = File(...)):
    ensure_ready()
    audio_path = await save_temp_audio(file)
    user_text = await asyncio.to_thread(transcribe_audio_sync, audio_path)
    async with history_lock:
        conversation_history.append({"role": "user", "text": user_text})
        if len(conversation_history) > MAX_TURNS * 2:
            del conversation_history[: len(conversation_history) - MAX_TURNS * 2]
        prompt = format_prompt(user_text)
    assistant_reply = await asyncio.to_thread(llm_generate_sync, prompt)
    async with history_lock:
        conversation_history.append({"role": "assistant", "text": assistant_reply})
        if len(conversation_history) > MAX_TURNS * 2:
            del conversation_history[: len(conversation_history) - MAX_TURNS * 2]
        memory_snapshot = list(get_recent_history())
    await asyncio.to_thread(save_chat_transcript_sync)
    tts_path = await asyncio.to_thread(synthesize_speech_sync, assistant_reply, file.filename)
    return JSONResponse(
        content={
            "transcription": user_text,
            "response": assistant_reply,
            "tts_audio": tts_path,
            "conversation_memory": memory_snapshot,
            "debug_prompt": prompt,
        }
    )

@app.get("/memory")
async def memory():
    async with history_lock:
        return get_recent_history()

@app.delete("/memory")
async def clear_memory():
    async with history_lock:
        conversation_history.clear()
    try:
        path = utils.get_file_path(TRANSCRIPT_FILE)
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass
    return {"cleared": True}

@app.get("/health")
def health():
    return {
        "ffmpeg_installed": check_ffmpeg_installed(),
        "asr_model_loaded": check_asr_model_loaded(),
        "llm_ready": check_llm_ready(),
        "gtts_ready": check_gtts_ready()
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
