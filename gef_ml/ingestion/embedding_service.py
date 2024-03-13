"""This module contains the EmbeddingService class, which is responsible for generating embeddings for a list of documents. The class uses the Together API to generate embeddings for the documents. The class also contains a method to generate embeddings in parallel for a list of documents. The method uses the aiolimiter library to limit the number of requests made to the Together API."""

import asyncio
import logging
import os
from typing import List

from tqdm.asyncio import tqdm_asyncio
from tqdm import tqdm
import aiohttp
import backoff
from aiolimiter import AsyncLimiter
from llama_index.core.schema import Document, MetadataMode

from gef_ml.utils.log_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(
        self,
        api_key: str | None = None,
        embed_endpoint_url: str = "https://api.together.xyz/v1/embeddings",
    ):
        self.api_key = api_key if api_key else os.getenv("TOGETHER_API_KEY")
        self.embed_endpoint_url = embed_endpoint_url

        if api_key is None and self.api_key is None:
            raise ValueError("TOGETHER_API_KEY environment variable must be set")

    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    @backoff.on_predicate(backoff.expo, lambda x: x is None, max_tries=3)
    async def _fetch_embeddings_with_retry(
        self,
        session: aiohttp.ClientSession,
        nodes: list[Document],
        model: str,
        embeddings_progress=None,
    ) -> list[Document]:

        req_input = [n.get_content(metadata_mode=MetadataMode.EMBED) for n in nodes]

        payload = {"input": req_input, "model": model}
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-type": "application/json",
        }

        async with session.post(
            self.embed_endpoint_url, json=payload, headers=headers
        ) as response:
            if response.status == 200:
                raw = await response.json()

                data = raw.get("data")

                if embeddings_progress:
                    embeddings_progress.update(1)

                for i, n in enumerate(nodes):
                    n.embedding = data[i]["embedding"]
                return nodes

            else:
                raise Exception(
                    f"Failed to generate embedding for node. HTTP Status: {response.status}. Reason: {response.reason}. Response: {await response.text()}\n\n Request: {payload}"
                )

    async def generate_embeddings(
        self,
        nodes: List[Document],
        model: str = "togethercomputer/m2-bert-80M-2k-retrieval",
        max_requests_per_second: int = 75,
        max_chunk_size: int = 10,
    ) -> List[Document]:
        timeout = aiohttp.ClientTimeout(total=60 * 60)  # 1 hour
        async with aiohttp.ClientSession(timeout=timeout) as session:
            limiter = AsyncLimiter(max_rate=max_requests_per_second, time_period=1)

            tasks = []

            chunks = [
                nodes[i : i + max_chunk_size]
                for i in range(0, len(nodes), max_chunk_size)
            ]

            for node_chunk in tqdm(chunks, desc="Creating embedding requests"):
                async with limiter:
                    task = asyncio.create_task(
                        self._fetch_embeddings_with_retry(session, node_chunk, model)
                    )
                    tasks.append(task)

            embeddings = await tqdm_asyncio.gather(*tasks, desc="Gathering embeddings")
            flattened_embeddings = [
                item for sublist in embeddings for item in sublist
            ]  # Flatten the list of lists

            return flattened_embeddings
