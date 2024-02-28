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
):
    if os.environ.get("TOGETHER_API_KEY") is None:
        raise ValueError("TOGETHER_API_KEY environment variable must be set")

    return IngestionPipeline(
        transformations=[
            SimpleFileNodeParser(),
            SentenceSplitter(chunk_size=384, chunk_overlap=64),
            TogetherEmbedding(
                model_name=together_embed_model_name,
                # api_key=os.environ.get["TOGETHER_API_KEY"],  # make sure this is set in the environment
            ),
        ],
        vector_store=vector_store,
    )
