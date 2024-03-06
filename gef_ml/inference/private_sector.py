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
from operator import inv
from typing import Literal, Optional, List

from llama_index.core import PromptHelper
from llama_index.core.response_synthesizers import TreeSummarize
from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.embeddings.together import TogetherEmbedding
from llama_index.llms.together import TogetherLLM
from pydantic import BaseModel, Field
from qdrant_client.http import models as qdrant_models

from gef_ml.utils import get_qdrant_vectorstore

logger = logging.getLogger(__name__)

QUERY_PRIVATE_SECTOR_INVOLVEMENT = """Determine instances of private sector involvement. Private sector involvement includes all of the following levels:

- 1: No private sector involvement - Projects which do not involve the private sector
- 2: Knowledge & Information Sharing - Projects in which the private sector is informed of initiatives and results
- 3: Policy Development - Projects where the private sector is consulted as part of an intervention run by someone else
- 4: Capacity Development - Projects focused on building the capacity of private sector actors, especially SMEs
- 5: Finance - Projects where government or civil society engages with private sector for finance and expertise
- 6: Industry Leadership - Projects that focus directly on the Private Sector as leader (private sector coming up with solutions; not only trained/capacitated)

If there is private sector involvement, please provide the level of involvement and the reason for the chosen involvement level."""


embed_model = TogetherEmbedding(model_name="togethercomputer/m2-bert-80M-2k-retrieval")
llm_model = TogetherLLM(model="mistralai/Mixtral-8x7B-Instruct-v0.1")

prompt_helper = PromptHelper(context_window=32768, num_output=512)


class ResponseObject(BaseModel):
    """Data model for the response to a private sector involvement query."""

    involvement_level: Literal[
        "No private sector involvement",
        "Knowledge & Information Sharing",
        "Policy Development",
        "Capacity Development",
        "Finance",
        "Industry Leadership",
    ] = Field(..., description="The level of private sector involvement")
    reason: str = Field(..., description="The reason for the chosen involvement level")


def retrieve_points(project_id: str, collection_name: str) -> List[NodeWithScore]:
    """Retrieve nodes from the vector store based on a project ID."""
    vector_store = get_qdrant_vectorstore(collection_name=collection_name)
    embed_model = TogetherEmbedding(
        model_name="togethercomputer/m2-bert-80M-2k-retrieval"
    )
    query_embedding = embed_model.get_query_embedding(QUERY_PRIVATE_SECTOR_INVOLVEMENT)

    qdrant_filters = qdrant_models.Filter(
        must=[
            qdrant_models.FieldCondition(
                key="project_id", match=qdrant_models.MatchValue(value=project_id)
            )
        ]
    )
    qdrant_query = VectorStoreQuery(query_embedding=query_embedding, similarity_top_k=5)

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


def determine_private_sector_involvement(
    project_id: str, qdrant_collection: str
) -> Optional[ResponseObject]:
    """Determine the level of private sector involvement for a given project ID."""
    nodes = retrieve_points(project_id, qdrant_collection)
    if not nodes:
        return None

    llm_model = TogetherLLM(model="mistralai/Mixtral-8x7B-Instruct-v0.1")
    prompt_helper = PromptHelper(context_window=32768, num_output=512)
    summarize = TreeSummarize(verbose=True, llm=llm_model, prompt_helper=prompt_helper, output_cls=ResponseObject)  # type: ignore

    response = summarize.synthesize(query=QUERY_PRIVATE_SECTOR_INVOLVEMENT, nodes=nodes)

    if isinstance(response, ResponseObject):
        logger.info(
            f"Private sector involvement for project_id {project_id}: {response.involvement_level}"
        )
        logger.info(f"Reason: {response.reason}")

        return response

    else:
        logger.error(
            f"Failed to determine private sector involvement for project_id {project_id}"
        )
        logger.info(f"Response: {response}")
        return None
