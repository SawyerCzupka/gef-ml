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
from typing import Literal, Optional, Union

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

# TODO find a better way to get the collection name
vector_store = get_qdrant_vectorstore(collection_name="temp")

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


def determine_private_sector_involvement(
    project_id: str,
) -> ResponseObject | None:
    query_str = QUERY_PRIVATE_SECTOR_INVOLVEMENT

    nodes = retrieve_points(project_id)

    if not nodes or len(nodes) == 0:
        logger.warning(f"No nodes found for project_id {project_id}")
        return None

    # TODO Use output class to represent the involvement levels to return a single one.
    summarize = TreeSummarize(
        verbose=True,
        llm=llm_model,
        prompt_helper=prompt_helper,
        output_cls=ResponseObject,  # type: ignore
    )

    response = summarize.synthesize(query=query_str, nodes=nodes)

    if isinstance(response, ResponseObject):  # Should always be true
        return response


def retrieve_points(project_id: str) -> list[NodeWithScore]:
    query_embedding = embed_model.get_query_embedding(QUERY_PRIVATE_SECTOR_INVOLVEMENT)

    # Filter Qdrant search by project_id
    qdrant_filters = qdrant_models.Filter(
        must=[
            qdrant_models.FieldCondition(
                key="project_id", match=qdrant_models.MatchValue(value=project_id)
            )
        ]
    )

    qdrant_query = VectorStoreQuery(
        query_embedding=query_embedding,
        similarity_top_k=5,
    )

    logger.info(
        f"Querying for project_id {project_id} in collection {vector_store.collection_name}"
    )
    query_results = vector_store.query(qdrant_query, qdrant_filters=qdrant_filters)

    nodes_with_scores = []
    if query_results.nodes is None:
        return nodes_with_scores

    logger.info(
        f"Found {len(query_results.nodes)} nodes for project_id {project_id} in collection {vector_store.collection_name}"
    )

    for index, node in enumerate(query_results.nodes):
        score: Optional[float] = None

        if query_results.similarities is not None:
            score = query_results.similarities[index]

        nodes_with_scores.append(NodeWithScore(node=node, score=score))

    return nodes_with_scores
