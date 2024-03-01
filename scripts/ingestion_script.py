from gef_ml.ingestion import StreamingIngestion
from gef_ml.utils import get_qdrant_vectorstore

from dotenv import load_dotenv

load_dotenv()


def main():
    vector_store = get_qdrant_vectorstore(collection_name="temp")

    ingest_manager = StreamingIngestion(
        directory="../data/to_ingest/", vector_store=vector_store
    )

    # Ingest the data

    ingest_manager.ingest()


if __name__ == "__main__":
    main()
