# voice_assistant/utils.py
import json
import os
import tempfile
from typing import Any

from fastapi import UploadFile

def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

def get_file_path(filename: str, folder: str | None = None) -> str:
    if folder:
        ensure_dir(folder)
        return os.path.join(folder, filename)
    return filename

def save_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

async def save_upload_to_temp(file: UploadFile, accepted_types: set[str]) -> str:
    if file.content_type not in accepted_types and not file.filename.lower().endswith(
        (".mp3", ".wav", ".m4a", ".ogg", ".webm")
    ):
        raise ValueError(f"Unsupported file format: {file.content_type}")
    suffix = os.path.splitext(file.filename)[1] or ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        data = await file.read()
        tmp.write(data)
        return tmp.name
