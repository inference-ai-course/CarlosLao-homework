# voice_assistant/memory.py
import asyncio
from typing import List, Dict

class Memory:
    def __init__(self, max_turns: int = 5):
        self._lock = asyncio.Lock()
        self._messages: List[Dict[str, str]] = []
        self._max_turns = max_turns

    async def append(self, role: str, text: str) -> None:
        async with self._lock:
            self._messages.append({"role": role, "text": text})
            # cap to 2 * max_turns messages
            overflow = len(self._messages) - 2 * self._max_turns
            if overflow > 0:
                del self._messages[:overflow]

    async def snapshot(self) -> List[Dict[str, str]]:
        async with self._lock:
            return list(self._messages[-2 * self._max_turns:])

    async def clear(self) -> int:
        async with self._lock:
            n = len(self._messages)
            self._messages.clear()
            return n
