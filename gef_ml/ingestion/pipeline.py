import logging

from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.vector_stores.types import BasePydanticVectorStore

# Configure logger

logger = logging.getLogger(__name__)


def get_pipeline(
    vector_store: BasePydanticVectorStore | None = None,
    together_embed_model_name: str = "togethercomputer/m2-bert-80M-32k-retrieval",
) -> IngestionPipeline:
    """
    Initializes and returns an ingestion pipeline with predefined transformations.
    """
    logger.debug(
        "Initializing ingestion pipeline with model: %s", together_embed_model_name
    )
    transformations = [
        SentenceSplitter(chunk_size=512, chunk_overlap=64, include_metadata=True),
    ]

    return IngestionPipeline(transformations=transformations)  # type: ignore
