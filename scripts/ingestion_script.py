from gef_ml.ingestion import StreamingIngestion
from gef_ml.utils import get_qdrant_vectorstore

from dotenv import load_dotenv

import logging
import asyncio

logging.basicConfig(level=logging.INFO)
load_dotenv()


async def main():
    vector_store = get_qdrant_vectorstore(collection_name="gef_6_512_64")

    ingest_manager = StreamingIngestion(
        directory="../data/gef-6/", vector_store=vector_store
    )

    # Ingest the data

    await ingest_manager.ingest()


if __name__ == "__main__":
    asyncio.run(main())
