import json
import logging
import os
import subprocess
import tempfile
import wave
from contextlib import asynccontextmanager, closing
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from transformers import pipeline
from dotenv import load_dotenv
import whisper
from gtts import gTTS
import utils

load_dotenv()
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
TRANSCRIPT_FILE = os.getenv("TRANSCRIPT_FILE", "transcript.txt")
RESPONSE_FOLDER = os.getenv("RESPONSE_FOLDER", "response")

if not HUGGINGFACE_TOKEN:
    raise RuntimeError("Missing Hugging Face token. Please set HUGGINGFACE_TOKEN.")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("voice-assistant")

asr_model = None
llm = None
conversation_history = []
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

@asynccontextmanager
async def lifespan(app: FastAPI):
    global asr_model, llm
    asr_model = whisper.load_model("small")
    llm = pipeline("text-generation", model="meta-llama/Llama-3.1-8B-Instruct", token=HUGGINGFACE_TOKEN)
    utils.get_script_dir("response")
    yield

app = FastAPI(title="Voice Assistant", version="2.0", lifespan=lifespan)

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
        logger.warning("Unsupported file format: %s (%s)", file.filename, file.content_type)
        raise HTTPException(400, "Unsupported file format")
    suffix = os.path.splitext(file.filename)[1] or ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        data = await file.read()
        tmp.write(data)
        logger.info("Saved uploaded audio to %s (%d bytes)", tmp.name, len(data))
        return tmp.name

def transcribe_audio(audio_path: str) -> str:
    logger.info("Transcribing %s", audio_path)
    try:
        result = asr_model.transcribe(audio_path, verbose=False)
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
        except:
            pass

def generate_response(user_prompt: str) -> str:
    conversation_history.append({"role": "user", "text": user_prompt})
    if len(conversation_history) > MAX_TURNS * 2:
        conversation_history[:] = conversation_history[-MAX_TURNS * 2:]
    prompt = "\n".join(f"{t['role']}: {t['text']}" for t in conversation_history) + "\nassistant:"
    logger.info("Generating response. Prompt length: %d", len(prompt))
    outputs = llm(prompt, max_new_tokens=150, do_sample=True, temperature=0.7)
    full_text = outputs[0]["generated_text"]
    reply = full_text[len(prompt):].strip().split("\n")[0]
    conversation_history.append({"role": "assistant", "text": reply})
    logger.info("Generated reply. Characters: %d", len(reply))
    return reply

def save_chat_transcript() -> str:
    path = utils.get_file_path("transcript.json")
    recent_turns = conversation_history[-MAX_TURNS * 2:]
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(recent_turns, f, indent=2, ensure_ascii=False)
        logger.info("Saved transcript as JSON with last %d turns to %s", MAX_TURNS, path)
    except Exception as e:
        logger.error("Failed to save JSON transcript: %s", e)
    return path

def synthesize_speech(text: str, input_filename: str) -> str:
    base_name = os.path.splitext(os.path.basename(input_filename))[0] or "response"
    out_path = utils.get_file_path(f"{base_name}_response.mp3", RESPONSE_FOLDER)
    try:
        logger.info("Synthesizing TTS to %s", out_path)
        gTTS(text=text, lang="en").save(out_path)
        return out_path
    except Exception as e:
        logger.error("TTS failed: %s", e)
        raise RuntimeError("TTS failed")

@app.post("/chat/")
async def chat(file: UploadFile = File(...)):
    logger.info("Received /chat request.")
    ensure_ready()
    audio_path = await save_temp_audio(file)
    user_text = transcribe_audio(audio_path)
    assistant_reply = generate_response(user_text)
    save_chat_transcript()
    tts_path = synthesize_speech(assistant_reply, file.filename)
    logger.info("Completed /chat request.")
    return JSONResponse(content={
        "transcription": user_text,
        "response": assistant_reply,
        "tts_audio": tts_path
    })

@app.get("/health")
def health():
    logger.info("Health check requested.")
    return {
        "ffmpeg_installed": check_ffmpeg_installed(),
        "asr_model_loaded": check_asr_model_loaded(),
        "llm_ready": check_llm_ready(),
        "gtts_ready": check_gtts_ready()
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    logger.info("Starting server on 0.0.0.0:%d", port)
    uvicorn.run(app, host="0.0.0.0", port=port)
