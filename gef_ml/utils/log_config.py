# log_config.py
import logging
import os


def setup_logging():
    log_file_path = os.path.join("/home/sawyer/git/gef-ml/", "ingestion.log")

    # Create a custom logger
    logger = logging.getLogger()
    logger.setLevel(
        logging.DEBUG
    )  # Set root logger to handle DEBUG and higher level logs

    # Create handlers
    c_handler = logging.StreamHandler()  # Console handler
    f_handler = logging.FileHandler(log_file_path)  # File handler

    # Create formatters and add it to handlers
    c_format = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
    f_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)

    # Set levels for handlers
    c_handler.setLevel(logging.INFO)  # Console handler for INFO and above
    f_handler.setLevel(logging.DEBUG)  # File handler for DEBUG and above

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)
