# config.py
"""Central configuration for the ML Q&A pipeline.

This module loads environment variables (from a `.env` file if present) and
defines all configuration constants used across the ArXiv fetcher, Q&A
generator, fine-tuning, and evaluation scripts. Paths, model parameters,
and hyperparameters are centralized here for consistency and maintainability.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from a .env file if present
load_dotenv()


class Config:
    """Holds configuration values for the ML Q&A pipeline.

    All attributes are class-level constants, loaded from environment variables
    with sensible defaults. Paths are created if they do not exist.
    """

    # ----------------------------
    # Base directories
    # ----------------------------
    BASE_DIR = Path(__file__).resolve().parent  # Root directory of the project
    DATA_DIR = BASE_DIR / os.getenv("DATA_FOLDER", "data")  # Data storage directory
    DATA_DIR.mkdir(exist_ok=True)

    # ----------------------------
    # Data file paths
    # ----------------------------
    OUTPUT_FILE = DATA_DIR / os.getenv(
        "ARXIV_OUTPUT_FILE", "arxiv_summaries.json"
    )  # ArXiv summaries output
    QA_INPUT_FILE = DATA_DIR / os.getenv(
        "QA_INPUT_FILE", "summaries_qa.json"
    )  # Q&A pairs input
    SYNTH_QA_FILE = DATA_DIR / os.getenv(
        "SYNTHETIC_QA_FILE", "synthetic_qa.jsonl"
    )  # Synthetic Q&A dataset output

    # ----------------------------
    # Dataset and ArXiv fetch parameters
    # ----------------------------
    TEXT_FIELD = os.getenv("DATASET_TEXT_FIELD", "text")  # Dataset text field name
    SEARCH_QUERY = os.getenv("ARXIV_QUERY", "machine learning")  # ArXiv search query
    MAX_RESULTS = int(
        os.getenv("ARXIV_MAX_RESULTS", "100")
    )  # Max number of ArXiv results
    GROUP_PREFIX = os.getenv(
        "ARXIV_ACTION_TEXT", "Summary Group:"
    )  # Prefix for grouped summaries
    GROUP_SIZE = int(os.getenv("SUMMARY_GROUP_SIZE", "5"))  # Papers per summary group

    # ----------------------------
    # Model directories and names
    # ----------------------------
    MODEL_DIR_BASE = (
        BASE_DIR / os.getenv("MODEL_OUTPUT_BASE", "models")
    ).resolve()  # Base directory for models
    MODEL_DIR_BASE.mkdir(parents=True, exist_ok=True)
    BASE_MODEL = os.getenv(
        "BASE_MODEL_ID", "unsloth/llama-3.1-8b-unsloth-bnb-4bit"
    )  # Base model ID or path
    MODEL_NAME = os.getenv(
        "MODEL_OUTPUT_NAME", "llama-3.1-8b-qlora-finetuned"
    )  # Fine-tuned model name
    MODEL_DIR = MODEL_DIR_BASE / MODEL_NAME  # Fine-tuned model directory
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # Training hyperparameters
    # ----------------------------
    LR = float(os.getenv("LEARNING_RATE", "5e-5"))  # Learning rate
    BATCH = int(os.getenv("BATCH_SIZE", "8"))  # Per-device batch size
    EPOCHS = int(os.getenv("NUM_EPOCHS", "5"))  # Number of training epochs
    GRAD_ACCUM = int(os.getenv("GRAD_ACCUM_STEPS", "4"))  # Gradient accumulation steps
    LOG_STEPS = int(os.getenv("LOGGING_STEPS", "10"))  # Logging interval (steps)

    FP16 = os.getenv("USE_FP16", "False").lower() == "true"  # Use FP16 precision
    BF16 = os.getenv("USE_BF16", "True").lower() == "true"  # Use BF16 precision

    # ----------------------------
    # LoRA parameters
    # ----------------------------
    LORA_R = int(os.getenv("LORA_R", "16"))  # LoRA rank
    LORA_ALPHA = int(os.getenv("LORA_ALPHA", "16"))  # LoRA alpha
    LORA_DROPOUT = float(os.getenv("LORA_DROPOUT", "0.0"))  # LoRA dropout
    USE_RSLORA = (
        os.getenv("USE_RSLORA", "False").lower() == "true"
    )  # Use rank-stabilized LoRA
    LORA_TARGETS = [  # Target modules for LoRA
        m.strip()
        for m in os.getenv(
            "LORA_TARGET_MODULES",
            "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        ).split(",")
        if m.strip()
    ]

    # ----------------------------
    # Gradient checkpointing
    # ----------------------------
    _gc = os.getenv("GRADIENT_CHECKPOINTING", "unsloth").strip().lower()
    if _gc in ("true", "1", "yes"):
        GRAD_CHECKPOINT = True
    elif _gc in ("false", "0", "no"):
        GRAD_CHECKPOINT = False
    else:
        GRAD_CHECKPOINT = "unsloth"  # Let the framework decide

    # ----------------------------
    # Evaluation parameters
    # ----------------------------
    EVAL_QA_FILE = DATA_DIR / os.getenv(
        "EVAL_QA_FILE", "evaluation_qa.jsonl"
    )  # Evaluation Q&A file
    EVAL_OUTPUT_FILE = DATA_DIR / os.getenv(
        "EVAL_OUTPUT_FILE", "evaluation_comparison.json"
    )  # Evaluation results file
    EVAL_PROMPT = os.getenv(  # System prompt for evaluation
        "EVAL_SYSTEM_PROMPT",
        "You are a helpful academic Q&A assistant specialized in scholarly content.",
    )
    EVAL_MAX_TOKENS = int(
        os.getenv("EVAL_MAX_NEW_TOKENS", "150")
    )  # Max tokens for evaluation output
