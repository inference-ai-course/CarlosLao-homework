"""
memory.py
---------
Conversation memory management module.

Provides MemoryService for storing and retrieving recent dialogue turns
between the user and assistant. Memory is kept in a rolling window of the
most recent exchanges, ensuring context is preserved without excessive growth.
"""

import json
import asyncio
from typing import List, Dict
from .config import MAX_TURNS, TRANSCRIPT_FILE
from .utils import get_file_path

class MemoryService:
    """
    Service for managing conversation memory state in a thread-safe manner.
    """
    def __init__(self):
        """
        Initialize with empty history and an asyncio lock
        to ensure safe concurrent access.
        """
        self.history: List[Dict[str, str]] = []
        self._lock = asyncio.Lock()

    async def add_user(self, text: str, speaker_id: str) -> None:
        """
        Add a user's message to the conversation history.
        """
        async with self._lock:
            self.history.append({"role": "user", "text": text, "speaker": speaker_id})
            self._trim()

    async def add_assistant(self, text: str) -> None:
        """
        Add the assistant's reply to the conversation history.
        """
        async with self._lock:
            self.history.append({"role": "assistant", "text": text})
            self._trim()

    def _trim(self) -> None:
        """
        Keep only the most recent MAX_TURNS user-assistant pairs.
        """
        max_len = MAX_TURNS * 2
        if len(self.history) > max_len:
            del self.history[: len(self.history) - max_len]

    async def get_recent(self) -> List[Dict[str, str]]:
        """
        Retrieve the most recent exchanges in the conversation.
        """
        async with self._lock:
            return list(self.history[-(2 * MAX_TURNS):])

    async def clear(self) -> None:
        """
        Clear all conversation history from memory.
        """
        async with self._lock:
            self.history.clear()

    async def save_transcript(self) -> str:
        """
        Save the most recent conversation turns to the transcript file.
        Returns the path to the file for reference.
        """
        path = get_file_path(TRANSCRIPT_FILE)
        async with self._lock:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(
                    self.history[-(2 * MAX_TURNS):],
                    f,
                    indent=2,
                    ensure_ascii=False
                )
        return path
