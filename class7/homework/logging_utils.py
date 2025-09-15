# logging_utils.py
"""Centralized logging configuration utilities.

This module provides a helper function to configure Python's logging system
with a consistent format, timestamp, and output stream. It is intended to be
imported and used by other scripts in the project to ensure uniform logging
behavior.
"""

import inspect
import logging
import sys


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure and return a logger for the calling module.

    This function sets up the root logger with a standard format that includes
    the timestamp, log level, logger name, and message. It uses a stream
    handler to output logs to standard output and forces reconfiguration to
    override any existing logging setup.

    The logger returned is specific to the module that called this function,
    allowing log messages to be tagged with the correct module name.

    Args:
        level (int, optional): Logging level to set for the root logger.
            Defaults to ``logging.INFO``.

    Returns:
        logging.Logger: A logger instance for the calling module.
    """
    caller_frame = inspect.stack()[1]
    caller_module = inspect.getmodule(caller_frame[0])
    caller_name = caller_module.__name__ if caller_module else "__main__"

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)-8s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

    return logging.getLogger(caller_name)
