"""
This file contains the logic for parsing, chunking, and loading data into a vector DB.

Functions:
- parse_data: Parses the input data and returns a list of chunks.
- chunk_data: Divides the parsed data into smaller chunks.
- load_data: Loads the chunks into a vector DB.
"""

import logging
import os

from llama_index.core import SimpleDirectoryReader
from llama_index.core.data_structs import BasePydanticVectorStore
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter, SimpleFileNodeParser
from llama_index.embeddings.together import TogetherEmbedding
from tqdm import tqdm

logger = logging.getLogger(__name__)


def get_pipeline(
    vector_store: BasePydanticVectorStore | None,
    together_embed_model_name: str = "togethercomputer/m2-bert-80M-32k-retrieval",
) -> IngestionPipeline:
    if os.environ.get("TOGETHER_API_KEY") is None:
        raise ValueError("TOGETHER_API_KEY environment variable must be set")

    transformations = [
        SimpleFileNodeParser(),
        SentenceSplitter(
            chunk_size=384, chunk_overlap=64
        ),  # Ensures that the chunks are not too large
        TogetherEmbedding(
            model_name=together_embed_model_name,
        ),
    ]

    return IngestionPipeline(
        transformations=transformations,
        vector_store=vector_store,
    )


class StreamingIngestion:
    """
    This class is used to stream data into the vector DB in chunks as opposed to loading every document in the directory and calling the pipeline on the whole thing. Within each chunk, the documents are loaded in parallel.

    This class assumes that the data is in a directory with subdirectories for each project. Each subdirectory contains the files for each project.
    """

    WORKERS_PER_CHUNK = 4

    def __init__(self, vector_store: BasePydanticVectorStore | None = None):
        self.vector_store = vector_store
        self.pipeline = get_pipeline(vector_store)

    def _ingest_project_id(self, project_id: str, show_progress: bool = True):
        """
        Ingests the data for a single project.
        """

        project_dir = os.path.join(
            self.directory, project_id
        )  # Get the project ID subdirectory
        reader = SimpleDirectoryReader(project_dir)  # Reads the project ID subdirectory

        documents = reader.load_data()  # Load the documents

        logger.debug(f"Loaded {len(documents)} documents for project {project_id}.")

        self.pipeline.run(
            show_progress=show_progress,
            documents=documents,
            num_workers=self.WORKERS_PER_CHUNK,
        )  # import directly to vector store

    def ingest_directory(self, directory: str):
        """
        Ingests the data for the entire directory.

        Args:
        - directory: The directory containing the data.
        """

        project_ids = [
            f
            for f in os.listdir(directory)
            if os.path.isdir(os.path.join(directory, f))
        ]

        logger.info(f"Found {len(project_ids)} projects in directory {directory}.")

        for project_id in tqdm(project_ids):
            logger.info(f"Ingesting project {project_id}: Started...")
            try:
                self._ingest_project_id(project_id, show_progress=False)
                logger.info(f"Ingesting project {project_id}: Done.")
            except Exception as e:
                logger.error(f"Ingesting project {project_id}: Failed.")
                logger.error(e)
