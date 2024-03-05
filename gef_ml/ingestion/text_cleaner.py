import logging
import re
import unicodedata

from llama_index.core.schema import Document, TransformComponent

from gef_ml.utils.log_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


class TextCleaner(TransformComponent):
    """
    Transform component to clean text extracted from documents.
    Things that should be cleaned include special characters like unicode characters and
    other non-standard whitespace or control characters.
    """

    def __call__(self, nodes: list[Document], **kwargs):
        for node in nodes:
            node.text = re.sub(r"(\w)\u00a0(\w)", r"\1 \2", node.text)
            node.text = re.sub(r"(\w)\u00a0", r"\1 ", node.text)
            node.text = re.sub(r"\u00a0(\w)", r" \1", node.text)

            # Replace non-breaking spaces with regular spaces
            node.text = node.text.replace("\u00a0", " ")
            # Replace other types of control characters
            node.text = re.sub(r"[\x00-\x1F\x7F-\x9F]", " ", node.text)
            # Normalize unicode to ensure consistent character representation
            node.text = unicodedata.normalize("NFKC", node.text)
            # Optionally, you might want to strip leading and trailing whitespace
            node.text = node.text.strip()

        return nodes
