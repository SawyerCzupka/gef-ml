from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams


def get_qdrant_vectorstore(
    collection_name: str, qdrant_client: QdrantClient | None = None
) -> QdrantVectorStore:
    """Get a QdrantVectorStore instance for a given collection name. If a QdrantClient is not provided, a new one will be created."""

    if qdrant_client is None:
        qdrant_client = QdrantClient("http://localhost:6333")

    vector_store = QdrantVectorStore(
        client=qdrant_client, collection_name=collection_name
    )

    return vector_store
