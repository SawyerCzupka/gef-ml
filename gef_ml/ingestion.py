"""
This file contains the logic for parsing, chunking, and loading data into a vector DB.

Functions:
- parse_data: Parses the input data and returns a list of chunks.
- chunk_data: Divides the parsed data into smaller chunks.
- load_data: Loads the chunks into a vector DB.
"""

from llama_index.core.ingestion import IngestionPipeline
from llama_index.embeddings.together import TogetherEmbedding
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SimpleFileNodeParser
from llama_index.core.node_parser import SentenceSplitter

from llama_index.core.data_structs import BasePydanticVectorStore

import os


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

    def __init__(
        self, directory: str, vector_store: BasePydanticVectorStore | None = None
    ):
        self.directory = directory
        self.vector_store = vector_store
        self.pipeline = get_pipeline(vector_store)

    def _ingest_project_id(self, project_id: str):
        """
        Ingests the data for a single project.
        """
        project_dir = os.path.join(
            self.directory, project_id
        )  # Get the project ID subdirectory
        reader = SimpleDirectoryReader(project_dir)  # Reads the project ID subdirectory
        
        documents = reader.