# voice_assistant/logging_config.py
import logging

def setup_logging(level: int | str = logging.INFO) -> logging.Logger:
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logger = logging.getLogger("voice_assistant")
    logger.setLevel(level)
    return logger
