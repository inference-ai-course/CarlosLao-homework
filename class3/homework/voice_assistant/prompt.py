# voice_assistant/prompt.py
import logging
from typing import List, Dict

logger = logging.getLogger("voice_assistant")

DEFAULT_SYSTEM = (
    "You are a concise, helpful voice assistant. "
    "Use the conversation history to stay in context. "
    "If information is missing, ask a brief clarifying question."
)

def normalize_role(role: str) -> str:
    r = role.lower().strip()
    if r in ("user", "human"):
        return "User"
    if r in ("assistant", "bot", "ai"):
        return "Assistant"
    if r == "system":
        return "System"
    return role.capitalize() or "User"

def format_prompt(user_text: str, history: List[Dict[str, str]], system_message: str = DEFAULT_SYSTEM) -> str:
    lines = [f"System: {system_message}", ""]
    for m in history:
        lines.append(f"{normalize_role(m['role'])}: {m['text'].strip()}")
    lines.append(f"User: {user_text.strip()}")
    lines.append("Assistant:")
    prompt = "\n".join(lines)
    logger.debug("Formatted prompt (first 300 chars): %s", prompt[:300])
    return prompt
