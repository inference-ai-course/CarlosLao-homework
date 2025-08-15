# Voice Assistant API

A lightweight FastAPI-based service for transforming short voice inputs into a full conversational experience. It includes speech transcription, response generation, and audio synthesis of replies.

---

## Features

- Speech-to-Text (ASR) using OpenAI Whisper
- Response Generation via Hugging Face transformers
- Text-to-Speech (TTS) with Google gTTS
- Conversation history management and transcript saving
- Health checks for external dependencies

---

## Project Structure

<PRE>
homework
├── prompt.txt                    # Prompt text template
├── readme.md                     # Project documentation
├── transcript.json               # Transcript data
├── generate_audio.py             # Script to generate audio files
├── main.py                       # Primary entry point
├── main_full.py                  # Extended/feature‑rich variant
├── main_plain.py                 # Minimal/plain variant
├── main_test.py                  # Test runner for main features
├── utils.py                      # Shared utility functions
│
├── audio/                        # Sample audio inputs
│   └── *.mp3
│
├── response/                     # Generated audio responses
│   └── *.mp3
│
├── voice_assistant/              # Core voice assistant package
│   ├── __init__.py
│   ├── asr.py                     # Speech recognition
│   ├── config.py                  # Configuration settings
│   ├── health.py                  # Health check endpoints
│   ├── llm.py                     # Language model integration
│   ├── logging_config.py          # Logging setup
│   ├── memory.py                  # Memory management
│   ├── prompt.py                  # Prompt building logic
│   ├── tts.py                     # Text‑to‑speech
│   ├── utils.py                   # Package utilities
│   │
│   └── routers/                   # API route handlers
│       ├── __init__.py
│       ├── chat.py
│       ├── health.py
│       └── memory.py
│
└── web/                           # Web frontend
    └── index.html
</PRE>

---

## Requirements

- Python 3.8 or higher
- FFmpeg installed and available on system PATH
- Hugging Face API token for model inference
- Internet access (required for gTTS)

---

## Setup

Clone and prepare your environment:

```bash
git clone <repository-url>
cd voice-assistant
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```
