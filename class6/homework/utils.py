import os
import sys
from typing import Optional

_base_dir: Optional[str] = None  # Declare at module level

def add_root_dir():
    global _base_dir
    try:
        _base_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        _base_dir = os.getcwd()

    if _base_dir not in sys.path:
        sys.path.insert(0, _base_dir)

def get_root_dir() -> str:
    global _base_dir
    if _base_dir is None:
        add_root_dir()
    return _base_dir

def get_script_dir(sub_dir: Optional[str] = None) -> str:
    """
    Returns the absolute path to the directory of the script.
    Optionally appends and creates a subdirectory.
    """
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        base_dir = os.getcwd()

    target_dir = os.path.join(base_dir, sub_dir) if sub_dir else base_dir
    os.makedirs(target_dir, exist_ok=True)
    return target_dir

def get_file_path(filename: str, sub_dir: Optional[str] = None) -> str:
    """
    Returns the full path of a file located in the script's directory.
    """
    return os.path.join(get_script_dir(sub_dir), filename)
