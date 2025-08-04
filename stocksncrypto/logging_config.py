import logging
import os

def setup_logging(log_level: str = "INFO", log_file: str = "crypto_tracker.log"):
    """Simple logging setup for both file and console."""
    # Remove all handlers associated with the root logger object.
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # File handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8', errors='replace')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    return root_logger
