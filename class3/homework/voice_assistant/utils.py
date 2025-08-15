"""
utils.py
--------
Shared utility functions for file handling, directories, and MIME type detection.
"""

import os
import mimetypes

def ensure_dir(path: str) -> str:
    """
    Create a directory if it doesn't already exist.
    """
    abs_path = os.path.abspath(path)
    os.makedirs(abs_path, exist_ok=True)
    return abs_path

def get_script_dir(relative: str | None = None) -> str:
    """
    Get the root directory of the project or a subdirectory within it.
    """
    base = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(base, ".."))
    if relative:
        full = os.path.join(root, relative)
        ensure_dir(full)
        return full
    return root

def get_file_path(filename: str, base_dir: str | None = None) -> str:
    """
    Build an absolute path to a file in the given base directory.
    """
    if base_dir:
        dir_abs = get_script_dir(base_dir)
        return os.path.join(dir_abs, filename)
    return os.path.join(get_script_dir(), filename)

def guess_media_type(path: str) -> str:
    """
    Guess the MIME type of a file based on its extension.
    Falls back to common audio types when mimetypes can't detect it.
    """
    # First try the standard library's best guess
    ctype, _ = mimetypes.guess_type(path)
    if ctype:
        return ctype

    # Manual fallbacks for common audio formats
    lower_path = path.lower()
    if lower_path.endswith(".mp3"):
        return "audio/mpeg"
    if lower_path.endswith(".wav"):
        return "audio/wav"
    if lower_path.endswith(".ogg"):
        return "audio/ogg"
    if lower_path.endswith(".m4a"):
        return "audio/x-m4a"
    if lower_path.endswith(".webm"):
        return "audio/webm"
    if lower_path.endswith(".flac"):
        return "audio/flac"
    if lower_path.endswith(".aac"):
        return "audio/aac"

    # Generic binary fallback
    return 
