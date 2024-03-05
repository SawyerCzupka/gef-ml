from gef_ml.ingestion import StreamingIngestion
from gef_ml.utils import get_qdrant_vectorstore
from gef_ml.utils.log_config import setup_logging

import logging
import asyncio

setup_logging()

logger = logging.getLogger(__name__)


async def main():
    vector_store = get_qdrant_vectorstore(collection_name="gef_6_512_64")

    ingest_manager = StreamingIngestion(
        directory="../data/gef-6/", vector_store=vector_store
    )

    await ingest_manager.ingest()


if __name__ == "__main__":
    asyncio.run(main())
