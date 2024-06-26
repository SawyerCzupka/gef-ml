"""
This file contains the logic determining whether a given project has involvement with the private sector.

Levels of private sector involvement:
- 1: No private sector involvement - Projects which do not involve the private sector
- 2: Knowledge & Information Sharing - Projects in which the private sector is informed of initiatives and results
- 3: Policy Development - Projects where the private sector is consulted as part of an intervention run by someone else
- 4: Capacity Development - Projects focused on building the capacity of private sector actors, especially SMEs
- 5: Finance - Projects where government or civil society engages with private sector for finance and expertise
- 6: Industry Leadership - Projects that focus directly on the Private Sector as leader (private sector coming up with solutions; not only trained/capacitated)

Also have the in depth descriptions from Eki
"""

import logging
import os
from typing import List, Literal, Optional

from llama_index.core import PromptHelper, VectorStoreIndex
from llama_index.core.base.response.schema import PydanticResponse
from llama_index.core.response_synthesizers import TreeSummarize
from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from llama_index.embeddings.together import TogetherEmbedding
from llama_index.llms.together import TogetherLLM
from pydantic import BaseModel, Field
from qdrant_client.http import models as qdrant_models

from gef_ml.utils import get_qdrant_vectorstore

from .prompts import (
    QUERY_PRIVATE_SECTOR_INVOLVEMENT,
    QUERY_PRIVATE_SECTOR_SUMMARY,
    QUERY_PRIVATE_SECTOR_SUMMARY_V2,
    PrivSectorClassResponseObj,
    PrivSectorSummaryResponseObj,
)

logger = logging.getLogger(__name__)

MAX_RETRIES = 3


if os.environ.get("TOGETHER_API_KEY") is None:
    raise ValueError("TOGETHER_API_KEY environment variable is not set")


def retrieve_points(
    project_id: str, collection_name: str, top_k: int = 20, mmr_threshold: float = 0.7
) -> List[NodeWithScore]:
    """Retrieve nodes from the vector store based on a project ID."""
    embed_model = TogetherEmbedding(
        model_name="togethercomputer/m2-bert-80M-2k-retrieval"
    )

    vector_store = get_qdrant_vectorstore(collection_name=collection_name)
    query_embedding = embed_model.get_query_embedding(QUERY_PRIVATE_SECTOR_INVOLVEMENT)

    qdrant_filters = qdrant_models.Filter(
        must=[
            qdrant_models.FieldCondition(
                key="project_id", match=qdrant_models.MatchValue(value=project_id)
            )
        ]
    )

    qdrant_query = VectorStoreQuery(
        query_embedding=query_embedding,
        similarity_top_k=top_k,
        mode=VectorStoreQueryMode.MMR,
        mmr_threshold=mmr_threshold,  # TODO: Determine the best threshold, closer to one is more similar, closer to zero is more diverse
    )

    logger.info(
        f"Querying for project_id {project_id} in collection {vector_store.collection_name}"
    )
    query_results = vector_store.query(qdrant_query, qdrant_filters=qdrant_filters)

    if not query_results.nodes:
        logger.warning(f"No nodes found for project_id {project_id}")
        return []

    logger.info(f"Found {len(query_results.nodes)} nodes for project_id {project_id}")
    nodes_with_scores = [
        NodeWithScore(node=node, score=score)
        for node, score in zip(query_results.nodes, query_results.similarities or [])
    ]
    return nodes_with_scores


def get_pydantic_query_engine(output_cls: BaseModel):
    vector_store = get_qdrant_vectorstore(collection_name="gef_6_1024_96")
    llm = TogetherLLM(model="mistralai/Mixtral-8x7B-Instruct-v0.1")

    index = VectorStoreIndex.from_vector_store(vector_store)
    query_engine = index.as_query_engine(
        output_cls=output_cls, response_mode="tree_summarize", llm=llm
    )

    query_engine.synthesize


def determine_private_sector_involvement(
    project_id: str, qdrant_collection: str, retry_num: int = 0
) -> Optional[PrivSectorClassResponseObj]:
    """Determine the level of private sector involvement for a given project ID."""
    nodes = retrieve_points(project_id, qdrant_collection)
    if not nodes:
        return None

    llm_model = TogetherLLM(model="mistralai/Mixtral-8x7B-Instruct-v0.1")
    prompt_helper = PromptHelper(context_window=32768, num_output=512)
    summarize = TreeSummarize(verbose=True, llm=llm_model, prompt_helper=prompt_helper, output_cls=PrivSectorClassResponseObj)  # type: ignore

    try:
        response = summarize.synthesize(
            query=QUERY_PRIVATE_SECTOR_INVOLVEMENT, nodes=nodes
        )
    except Exception as e:
        logger.exception(f"Error synthesizing response for project_id {project_id}")
        if retry_num < MAX_RETRIES:
            logger.warning(f"Retrying for project_id {project_id}")
            return determine_private_sector_involvement(
                project_id, qdrant_collection, retry_num=retry_num + 1
            )
        else:
            logger.error(
                f"Failed to determine private sector involvement for project_id {project_id} within {retry_num} retries"
            )
            return None

    if isinstance(response, PydanticResponse):
        if isinstance(response.response, PrivSectorClassResponseObj):
            structured_response = PrivSectorClassResponseObj(
                involvement_level=response.response.involvement_level,
                secondary_involvement_level=response.response.secondary_involvement_level,
                reason=response.response.reason,
                extra_info=response.response.extra_info,
            )

    if structured_response:
        return structured_response

    else:
        logger.info(f"Didn't return the pydantic object but probably still worked")
        logger.info(f"Response: {response}")
        logger.info(f"Type: {type(response)}")
        return None


def generate_private_sector_summary(
    project_id: str, qdrant_collection: str
) -> Optional[PrivSectorSummaryResponseObj] | str:
    """Generate a summary of the private sector involvement for a given project ID."""
    nodes = retrieve_points(project_id, qdrant_collection, top_k=25, mmr_threshold=0.6)
    if not nodes:
        return None

    llm_model = TogetherLLM(model="mistralai/Mixtral-8x7B-Instruct-v0.1")
    prompt_helper = PromptHelper(context_window=32768, num_output=768)
    summarize = TreeSummarize(verbose=True, llm=llm_model, prompt_helper=prompt_helper)  # type: ignore   , output_cls=PrivSectorSummaryResponseObj

    try:
        response = summarize.synthesize(query=QUERY_PRIVATE_SECTOR_SUMMARY, nodes=nodes)
    except Exception as e:
        logger.exception(f"Error synthesizing response for project_id {project_id}")
        return None

    logger.info(f"Response: {response}")
    print(response)
    print(type(response))

    structured_response = None

    if isinstance(response, PydanticResponse):
        if isinstance(response.response, PrivSectorSummaryResponseObj):
            structured_response = PrivSectorSummaryResponseObj(
                summary=response.response.summary
            )

    if structured_response:
        return structured_response

    elif response:
        return response  # type: ignore

    else:
        logger.info(f"Didn't return the pydantic object but probably still worked")
        logger.info(f"Response: {response}")
        logger.info(f"Type: {type(response)}")
        return None
