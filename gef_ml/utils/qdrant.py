from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from llama_index.vector_stores.qdrant import QdrantVectorStore


def get_qdrant_vectorstore(
    collection_name: str, qdrant_client: QdrantClient | None = None
) -> QdrantVectorStore:
    if qdrant_client is None:
        qdrant_client = QdrantClient("http://localhost:6333")

    # qdrant_client.recreate_collection(
    #     collection_name=collection_name,
    #     vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    # )

    vector_store = QdrantVectorStore(
        client=qdrant_client, collection_name=collection_name
    )

    return vector_store
