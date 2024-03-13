"""This module contains the StreamingIngestion class, which is used to stream data into the vector DB in chunks as opposed to loading every document in the directory and calling the pipeline on the whole thing. Within each chunk, the documents are loaded in parallel."""

import logging
import os

from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import Document
from llama_index.core.vector_stores.types import BasePydanticVectorStore
from tqdm import tqdm

from gef_ml.ingestion.embedding_service import EmbeddingService
from gef_ml.ingestion.pipeline import get_pipeline
from gef_ml.utils import file_metadata
from gef_ml.utils.log_config import setup_logging

load_dotenv()
setup_logging()
logger = logging.getLogger(__name__)

embed_service = EmbeddingService()


class StreamingIngestion:
    """
    This class is used to stream data into the vector DB in chunks as opposed to loading every document in the directory and calling the pipeline on the whole thing. Within each chunk, the documents are loaded in parallel.

    This class assumes that the data is in a directory with subdirectories for each project. Each subdirectory contains the files for each project.
    """

    WORKERS_PER_CHUNK = 1

    def __init__(
        self,
        directory: str,
        vector_store: BasePydanticVectorStore | None = None,
        chunk_size=512,
        chunk_overlap=64,
    ):
        logger.info("Initializing StreamingIngestion for directory: %s", directory)
        self.directory = directory
        self.vector_store = vector_store
        self.pipeline = get_pipeline(
            vector_store,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            include_metadata=False,
        )

    def _ingest_project_id(
        self, project_id: str, show_progress: bool = True
    ) -> list[Document]:
        """
        Ingests documents for a given project ID, returning the processed nodes.
        """

        project_dir = os.path.join(self.directory, project_id)
        logger.info(
            "Ingesting documents for project ID: %s from %s", project_id, project_dir
        )
        reader = SimpleDirectoryReader(project_dir, file_metadata=file_metadata)
        documents = reader.load_data(num_workers=4)
        logger.info("Loaded %d documents for project %s.", len(documents), project_id)

        processed_nodes = self.pipeline.run(
            show_progress=show_progress,
            documents=documents,
            num_workers=self.WORKERS_PER_CHUNK,
        )
        logger.info(
            "Processed %d documents for project %s.", len(processed_nodes), project_id
        )

        return processed_nodes  # type: ignore

    async def ingest(self):
        """
        Ingests the data for the entire directory.

        Args:
        - directory: The directory containing the data.
        """

        project_ids = [
            f
            for f in os.listdir(self.directory)
            if os.path.isdir(os.path.join(self.directory, f))
        ]
        logger.info(
            "Found %d projects in directory %s", len(project_ids), self.directory
        )

        for project_id in tqdm(project_ids, desc="Ingesting projects"):
            logger.info("Starting ingestion for project %s", project_id)
            try:
                nodes = self._ingest_project_id(project_id, show_progress=True)
                embeddings = await embed_service.generate_embeddings(
                    nodes, max_chunk_size=8
                )
                logger.info(
                    "Adding embeddings to vector store for project %s", project_id
                )
                if self.vector_store:
                    self.vector_store.add(embeddings)  # type: ignore
                logger.info("Completed ingestion for project %s", project_id)
            except Exception as e:
                logger.error(
                    "Failed to ingest project %s due to error: %s", project_id, e
                )
                logger.exception("Error details:")
