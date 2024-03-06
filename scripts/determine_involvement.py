import logging

from dotenv import load_dotenv

load_dotenv()

from gef_ml.inference import determine_private_sector_involvement
from gef_ml.utils.log_config import setup_logging


setup_logging()

logger = logging.getLogger(__name__)

GEF_6_COLLECTION = "gef_6_512_64"


def main():
    project_id = "6973"
    collection = GEF_6_COLLECTION

    response = determine_private_sector_involvement(
        project_id, qdrant_collection=collection
    )

    if response:
        logger.info(f"Private sector involvement: {response.involvement_level}")
        logger.info(f"Reason: {response.reason}")


if __name__ == "__main__":

    main()
