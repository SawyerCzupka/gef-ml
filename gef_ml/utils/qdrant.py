from qdrant_client import QdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore


def get_qdrant_vectorstore(
    collection_name: str, qdrant_client: QdrantClient | None = None
) -> QdrantVectorStore:
    if qdrant_client is None:
        qdrant_client = QdrantClient("http://localhost:6333")

    vector_store = QdrantVectorStore(
        client=qdrant_client, collection_name=collection_name
    )

    return vector_store
