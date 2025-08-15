"""
prompt.py
---------
Prompt assembly utilities.

Contains helpers for formatting conversation history and system instructions
into a structured prompt suitable for passing to the language model.
"""

from typing import List, Dict
from .config import DEFAULT_SYSTEM_PROMPT

def normalize_role(role: str) -> str:
    """
    Map raw role strings to canonical role labels for prompt display.
    """
    role = (role or "").lower().strip()
    mapping = {"user": "User", "assistant": "Assistant", "system": "System"}
    return mapping.get(role, "User")

def format_prompt(
    history: List[Dict[str, str]],
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
    lines = [f"System: {system_message}", ""]
    for m in history:
        role = normalize_role(m.get("role"))
        text = (m.get("text") or "").strip()
        speaker = m.get("speaker")
        prefix = f"{role}({speaker})" if include_speaker and role == "User" and speaker else role
        lines.append(f"{prefix}: {text}")
    lines.append("Assistant:")
    return "\n".join(lines)
