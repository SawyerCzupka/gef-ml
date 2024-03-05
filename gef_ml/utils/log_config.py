import logging
import os
from colorlog import ColoredFormatter

_already_setup = False


class ExcludeSpecificLoggerFilter(logging.Filter):
    def __init__(self, name, level):
        super().__init__(name)
        self.level = level

    def filter(self, record):
        # Check if the record comes from the specified logger and its level is DEBUG
        if self.name in record.name and record.levelno == logging.DEBUG:
            return False  # Exclude DEBUG logs from the specified logger
        return True  # Include all other logs


def setup_logging():
    global _already_setup
    if _already_setup:
        return

    else:
        _already_setup = True

    log_file_path = os.path.join("/home/sawyer/git/gef-ml/", "ingestion.log")

    # Create a custom logger
    logger = logging.getLogger()
    logger.setLevel(
        logging.DEBUG
    )  # Set root logger to handle DEBUG and higher level logs

    # Define log format
    log_format = (
        "%(asctime)s - %(name)s - %(log_color)s%(levelname)s%(reset)s - %(message)s"
    )
    colors = {
        "DEBUG": "cyan",
        "INFO": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "red,bg_white",
    }

    # Console handler with colored output
    c_handler = logging.StreamHandler()
    c_format = ColoredFormatter(log_format, log_colors=colors)
    c_handler.setFormatter(c_format)
    c_handler.setLevel(logging.INFO)

    # File handler with standard formatting
    f_handler = logging.FileHandler(log_file_path)
    f_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    f_handler.setFormatter(f_format)
    f_handler.setLevel(logging.DEBUG)

    # Add a filter to exclude DEBUG logs from 'llamaindex'
    f_handler.addFilter(ExcludeSpecificLoggerFilter("llama_index", logging.DEBUG))

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)


__all__ = ["setup_logging"]
