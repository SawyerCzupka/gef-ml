"""
This file contains the logic for parsing, chunking, and loading data into a vector DB.

Functions:
- parse_data: Parses the input data and returns a list of chunks.
- chunk_data: Divides the parsed data into smaller chunks.
- load_data: Loads the chunks into a vector DB.
"""

import asyncio
import logging
import os
from typing import Sequence

import aiohttp
from aiolimiter import AsyncLimiter
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode
from llama_index.core.vector_stores.types import BasePydanticVectorStore

# Load environment variables
load_dotenv()

# Ensure required API key is set
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
if TOGETHER_API_KEY is None:
    raise ValueError("TOGETHER_API_KEY environment variable must be set")

EMBED_ENDPOINT_URL = "https://api.together.xyz/v1/embeddings"

# Configure logger
logger = logging.getLogger(__name__)


def get_pipeline(
    vector_store: BasePydanticVectorStore = None,
    together_embed_model_name: str = "togethercomputer/m2-bert-80M-32k-retrieval",
) -> IngestionPipeline:
    """
    Initializes and returns an ingestion pipeline with predefined transformations.
    """
    transformations = [
        SentenceSplitter(chunk_size=384, chunk_overlap=64, include_metadata=True),
    ]
    return IngestionPipeline(transformations=transformations)


def file_metadata(filename: str) -> dict[str, str]:
    """
    Extracts metadata from the given filename.
    """
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
    return project_id, doc_id


class StreamingIngestion:
    """
    This class is used to stream data into the vector DB in chunks as opposed to loading every document in the directory and calling the pipeline on the whole thing. Within each chunk, the documents are loaded in parallel.

    This class assumes that the data is in a directory with subdirectories for each project. Each subdirectory contains the files for each project.
    """

    WORKERS_PER_CHUNK = 1

    def __init__(
        self, directory: str, vector_store: BasePydanticVectorStore | None = None
    ):
        self.directory = directory
        self.vector_store = vector_store
        self.pipeline = get_pipeline(vector_store)

    def _ingest_project_id(
        self, project_id: str, show_progress: bool = True
    ) -> Sequence[BaseNode]:
        """
        Ingests documents for a given project ID, returning the processed nodes.
        """

        project_dir = os.path.join(self.directory, project_id)
        reader = SimpleDirectoryReader(project_dir, file_metadata=file_metadata)
        documents = reader.load_data(num_workers=4)
        logger.info(f"Loaded {len(documents)} documents for project {project_id}.")
        return self.pipeline.run(
            show_progress=show_progress,
            documents=documents,
            num_workers=self.WORKERS_PER_CHUNK,
        )

    async def fetch_embedding(
        self,
        session: aiohttp.ClientSession,
        node: BaseNode,
        model: str = "togethercomputer/m2-bert-80M-32k-retrieval",
    ) -> dict:
        payload = {"input": node.get_content(metadata_mode="all"), "model": model}
        headers = {
            "Authorization": f"Bearer {TOGETHER_API_KEY}",
            "Content-type": "application/json",
        }
        async with session.post(
            EMBED_ENDPOINT_URL, json=payload, headers=headers
        ) as response:
            if response.status == 200:
                data = await response.json()
                return data.get("embedding")
            else:
                logger.error(
                    f"Failed to generate embedding for node. Status: {response.status}"
                )
                return None

    async def generate_embeddings_rest(
        self,
        nodes: Sequence[BaseNode],
        model: str = "togethercomputer/m2-bert-80M-32k-retrieval",
        max_requests_per_second: int = 150,
    ) -> Sequence[BaseNode]:
        """
        Generate embeddings for a list of documents using the Together API.

        Args:
        - nodes: A list of documents to generate embeddings for.
        - model: The model to use for generating the embeddings.

        Returns:
        - A list of embeddings for the documents.
        """
        async with aiohttp.ClientSession() as session:
            tasks = []
            limiter = AsyncLimiter(max_rate=max_requests_per_second, time_period=1)
            for node in nodes:
                async with limiter:
                    task = asyncio.create_task(
                        self.fetch_embedding(session, node, model)
                    )
                    tasks.append(task)
            embeddings = await asyncio.gather(*tasks)
        for node, embedding in zip(nodes, embeddings):
            if embedding:
                node.embedding = embedding
        return nodes

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
        logger.info(f"Found {len(project_ids)} projects in directory {self.directory}.")

        for project_id in project_ids:
            logger.info(f"Ingesting project {project_id}: Started...")
            try:
                nodes = self._ingest_project_id(project_id, show_progress=True)
                embeddings = await self.generate_embeddings_rest(nodes)
                # Assuming self.vector_store has an add method to store embeddings
                if self.vector_store:
                    self.vector_store.add(embeddings)
                logger.info(f"Ingesting project {project_id}: Done.")
            except Exception as e:
                logger.error(f"Ingesting project {project_id}: Failed with error {e}")
