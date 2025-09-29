# config.py
"""Centralized configuration for the scientific summarization and reward modeling pipeline.

This module loads environment variables, applies defaults, and exposes a Config
class with attributes for data paths, model parameters, prompts, and training
hyperparameters. It ensures reproducibility and onboarding clarity by grouping
all configurable values in one place.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Base directories
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent

# Load environment variables from .env files
load_dotenv(PROJECT_ROOT / ".env")
load_dotenv(BASE_DIR / ".env")


def _get_int(name: str, default: int) -> int:
    """Retrieve an environment variable as an integer."""
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


def _get_float(name: str, default: float) -> float:
    """Retrieve an environment variable as a float."""
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


def _get_bool(name: str, default: str = "false") -> bool:
    """Retrieve an environment variable as a boolean."""
    return str(os.getenv(name, default)).strip().lower() in {"1", "true", "yes", "y"}


class Config:
    """Holds configuration values for all pipeline stages."""

    # Data paths
    DATA_DIR = BASE_DIR / os.getenv("DATA_FOLDER_NAME", "data")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    RAW_FOLDER = DATA_DIR / os.getenv("RAW_FOLDER_NAME", "raw")
    RAW_FOLDER.mkdir(parents=True, exist_ok=True)

    REWARD_DATA_FILE = DATA_DIR / os.getenv("REWARD_DATA_FILE", "reward_data.jsonl")

    # Model settings
    LLAMA_MODEL_NAME = os.getenv("LLAMA_MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
    HUGGINGFACE_HUB_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN", "")

    DEVICE = os.getenv("DEVICE", "auto")
    QUANT_MODE = os.getenv("QUANT_MODE", "8bit")
    COMPUTE_DTYPE = os.getenv("COMPUTE_DTYPE", "float16")

    MAX_NEW_TOKENS = _get_int("MAX_NEW_TOKENS", 150)
    DETERMINISTIC = _get_bool("DETERMINISTIC", "false")
    BATCH_SIZE = _get_int("BATCH_SIZE", 2)

    # Chunking and tokenization
    CHUNK_SIZE = _get_int("CHUNK_SIZE", 1000)
    MAX_TOKENS = _get_int("MAX_TOKENS", 4096)

    # ArXiv ingestion
    USE_ARXIV_ABSTRACTS = _get_bool("USE_ARXIV_ABSTRACTS", "true")
    ARXIV_QUERY = os.getenv("ARXIV_QUERY", "machine learning")
    MAX_RESULTS = _get_int("MAX_RESULTS", 10)

    # Prompts
    CHUNK_PROMPT = os.getenv(
        "CHUNK_PROMPT",
        "Summarize this section faithfully in 3-5 sentences.",
    )
    PROMPT_CHOSEN = os.getenv(
        "PROMPT_CHOSEN",
        "Write a 160-200 word faithful scientific summary focusing on methodology and results.",
    )
    PROMPT_REJECTED = os.getenv(
        "PROMPT_REJECTED",
        "Write a 160-200 word accessible summary highlighting motivation, novelty, and potential impact.",
    )

    # Generation parameters
    CHOSEN_TEMPERATURE = _get_float("CHOSEN_TEMPERATURE", 0.7)
    CHOSEN_TOP_K = _get_int("CHOSEN_TOP_K", 50)
    CHOSEN_TOP_P = _get_float("CHOSEN_TOP_P", 0.9)

    REJECTED_TEMPERATURE = _get_float("REJECTED_TEMPERATURE", 1.0)
    REJECTED_TOP_K = _get_int("REJECTED_TOP_K", 0)
    REJECTED_TOP_P = _get_float("REJECTED_TOP_P", 0.95)

    # Reward model training
    REWARD_MODEL_NAME = os.getenv("REWARD_MODEL_NAME", "microsoft/deberta-v3-base")
    REWARD_OUTPUT_DIR = BASE_DIR / os.getenv("REWARD_OUTPUT_DIR", "reward_model")
    REWARD_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    REWARD_MAX_LENGTH = _get_int("REWARD_MAX_LENGTH", 1024)
    REWARD_BATCH_SIZE = _get_int("REWARD_BATCH_SIZE", 8)
    REWARD_NUM_EPOCHS = _get_int("REWARD_NUM_EPOCHS", 3)
    REWARD_LEARNING_RATE = _get_float("REWARD_LEARNING_RATE", 5e-6)
    REWARD_FP16 = _get_bool("REWARD_FP16", "true")
    REWARD_SEED = _get_int("REWARD_SEED", 42)
    REWARD_LOGGING_STEPS = _get_int("REWARD_LOGGING_STEPS", 10)
    REWARD_SAVE_STRATEGY = os.getenv("REWARD_SAVE_STRATEGY", "epoch")

    @classmethod
    def as_dict(cls) -> dict:
        """Return configuration values as a dictionary.

        Returns:
            dict: Mapping of config keys to values, with Paths converted to strings.
        """
        result: dict = {}
        for k, v in cls.__dict__.items():
            if not k.startswith("_") and k != "as_dict":
                result[k] = str(v) if isinstance(v, Path) else v
        return result
