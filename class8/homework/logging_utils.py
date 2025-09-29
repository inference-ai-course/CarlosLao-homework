# logging_utils.py
"""Provides logging configuration and package installation utilities.

This module standardizes logging across the project and includes a helper
to install dependencies from a requirements file.
"""

import logging
import subprocess
import sys
from pathlib import Path

log = logging.getLogger(__name__)


def configure_logging(level=logging.INFO) -> logging.Logger:
    """Configure and return a logger.

    Args:
        level (int): Logging level (default: logging.INFO).

    Returns:
        logging.Logger: Configured logger instance.
    """
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s - %(levelname)-8s - %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[logging.StreamHandler(sys.stdout)],
        )
    return logging.getLogger(__name__)


def install_packages(requirements_filename: str = "requirements.txt") -> None:
    """Install packages from a requirements file.

    Args:
        requirements_filename (str): Name of the requirements file.

    Raises:
        FileNotFoundError: If the requirements file does not exist.
        RuntimeError: If package installation fails.
    """
    script_dir = Path(__file__).resolve().parent
    requirements_path = script_dir / requirements_filename
    if not requirements_path.exists():
        raise FileNotFoundError(f"Requirements file not found: {requirements_path}")

    log.info("Installing packages from %s", requirements_path)
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", str(requirements_path)]
        )
        log.info("All packages installed successfully.")
    except subprocess.CalledProcessError as e:
        log.error("Package installation failed: %s", e)
        raise RuntimeError("Package installation failed") from e
