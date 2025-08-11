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

```plaintext
voice-assistant/
├── main.py               # FastAPI app with endpoints
├── utils.py              # Helper functions for file handling
├── transcript.json       # Stores chat history
├── response/             # Folder for generated TTS audio files
├── .env                  # Environment variables
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

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
