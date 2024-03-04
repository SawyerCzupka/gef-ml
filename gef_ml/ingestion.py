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
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio


from gef_ml.utils.log_config import setup_logging

setup_logging()

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
    logger.debug(
        "Initializing ingestion pipeline with model: %s", together_embed_model_name
    )
    transformations = [
        SentenceSplitter(chunk_size=512, chunk_overlap=64, include_metadata=True),
    ]
    return IngestionPipeline(transformations=transformations)


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


class StreamingIngestion:
    """
    This class is used to stream data into the vector DB in chunks as opposed to loading every document in the directory and calling the pipeline on the whole thing. Within each chunk, the documents are loaded in parallel.

    This class assumes that the data is in a directory with subdirectories for each project. Each subdirectory contains the files for each project.
    """

    WORKERS_PER_CHUNK = 1

    def __init__(
        self, directory: str, vector_store: BasePydanticVectorStore | None = None
    ):
        logger.info("Initializing StreamingIngestion for directory: %s", directory)
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

        return processed_nodes

    async def fetch_embedding(
        self,
        session: aiohttp.ClientSession,
        node: BaseNode,
        model: str = "togethercomputer/m2-bert-80M-32k-retrieval",
        embeddings_progress=None,
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

                if embeddings_progress:
                    embeddings_progress.update(1)

                embedding = data["data"][0]["embedding"]

                return embedding
            else:
                logger.error(
                    "Failed to generate embedding for node. HTTP Status: %d",
                    response.status,
                )
                return None

    async def generate_embeddings_rest(
        self,
        nodes: Sequence[BaseNode],
        model: str = "togethercomputer/m2-bert-80M-32k-retrieval",
        max_requests_per_second: int = 100,
    ) -> Sequence[BaseNode]:
        """
        Generate embeddings for a list of documents using the Together API.

        Args:
        - nodes: A list of documents to generate embeddings for.
        - model: The model to use for generating the embeddings.

        Returns:
        - A list of embeddings for the documents.
        """
        logger.info(
            "Generating embeddings for %d nodes using model %s", len(nodes), model
        )
        async with aiohttp.ClientSession() as session:
            tasks = []
            limiter = AsyncLimiter(max_rate=max_requests_per_second, time_period=1)

            iterator = tqdm(nodes, desc="Creating tasks")
            embeddings_progress = tqdm_asyncio(nodes, desc="Generating embeddings")

            for node in iterator:
                async with limiter:
                    task = asyncio.create_task(
                        self.fetch_embedding(session, node, model, embeddings_progress)
                    )
                    tasks.append(task)
            # embeddings = await tqdm_asyncio.gather(*tasks, desc="Generating embeddings")

            embeddings = await asyncio.gather(*tasks)

        for node, embedding in zip(nodes, embeddings):
            if not isinstance(embedding, Exception):
                try:
                    node.embedding = embedding

                except Exception as e:
                    logger.error("Failed to set embedding for node due to error: %s", e)
                    logger.error("Embedding: %s", embedding)
                    logger.exception("Error details:")
            else:
                logger.error("Node embedding is an exception: %s", embedding)
                logger.error("Node: %s", node)
                logger.exception("Error details:")

        logger.info("Generated embeddings for %d nodes", len(nodes))
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
        logger.info(
            "Found %d projects in directory %s", len(project_ids), self.directory
        )

        for project_id in tqdm(project_ids, desc="Ingesting projects"):
            logger.info("Starting ingestion for project %s", project_id)
            try:
                nodes = self._ingest_project_id(project_id, show_progress=True)
                embeddings = await self.generate_embeddings_rest(nodes)
                # Assuming self.vector_store has an add method to store embeddings
                logger.info(
                    "Adding embeddings to vector store for project %s", project_id
                )
                if self.vector_store:
                    self.vector_store.add(embeddings)
                logger.info("Completed ingestion for project %s", project_id)
            except Exception as e:
                logger.error(
                    "Failed to ingest project %s due to error: %s", project_id, e
                )
                logger.exception("Error details:")
