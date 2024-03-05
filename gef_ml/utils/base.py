import logging
import os

from gef_ml.utils.log_config import setup_logging


setup_logging()

logger = logging.getLogger(__name__)


def file_metadata(filename: str) -> dict[str, str]:
    """
    Extracts metadata from the given filename.
    """
    logger.debug("Extracting metadata from filename: %s", filename)
    base_name = os.path.basename(filename)
    project_id, doc_id = parse_filename(base_name)
    return {
        "filename": base_name,
        "extension": os.path.splitext(base_name)[1],
        "project_id": project_id,
        "doc_id": doc_id,
    }


def parse_filename(filename: str) -> tuple[str, str]:
    """
    Parses the filename to extract project and document IDs.
    """
    parts = filename.split("_")
    project_id = parts[0][1:]
    doc_id = parts[1].split(".")[0][3:]
    logger.debug(
        "Parsed filename %s into project_id=%s, doc_id=%s", filename, project_id, doc_id
    )
    return project_id, doc_id


__all__ = ["file_metadata", "parse_filename"]
