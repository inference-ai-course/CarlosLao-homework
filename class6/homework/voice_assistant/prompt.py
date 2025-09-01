"""
prompt.py
---------
Prompt assembly utilities.

Contains helpers for formatting conversation history and system instructions
into a structured prompt suitable for passing to the language model.
"""

from typing import List, Dict
from .config import DEFAULT_SYSTEM_PROMPT
from .memory import Message

def normalize_role(role: str) -> str:
    """
    Map raw role strings to canonical role labels for prompt display.
    """
    role = (role or "").lower().strip()
    mapping = {"user": "User", "assistant": "Assistant", "system": "System"}
    return mapping.get(role, "User")

def format_prompt(
    history: List[Message],
    system_message: str = DEFAULT_SYSTEM_PROMPT,
    include_speaker: bool = True
) -> str:
    """
    Format conversation history into a prompt string.

    Parameters
    ----------
    history : list of dict
        The conversation history with role, text, and optional speaker.
    system_message : str
        The system-level instruction or persona definition.
    include_speaker : bool
        Whether to annotate user lines with their speaker_id.

    Returns
    -------
    str
        Combined, line-by-line conversation prompt text.
    """
    lines = []
    for msg in history:
        if msg.role == "user":
            lines.append(f"User({msg.speaker}): {msg.text}")
        elif msg.role == "assistant":
            lines.append(f"Assistant: {msg.text}")
    return "\n".join(lines)
