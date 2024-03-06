import logging

from dotenv import load_dotenv
import csv

load_dotenv()

from gef_ml.inference import determine_private_sector_involvement
from gef_ml.utils.log_config import setup_logging


setup_logging()

logger = logging.getLogger(__name__)

GEF_6_COLLECTION = "gef_6_512_64"

OUTPUT_CSV = "involvement_results.csv"


def determine_involvement_batch(
    project_ids: list[str], qdrant_collection: str = GEF_6_COLLECTION
):

    logger.info(f"Processing {len(project_ids)} project IDs")
    path = "../data/" + OUTPUT_CSV
    with open(path, "w", newline="") as f:
        logger.info(f"Writing results to {path}")
        writer = csv.writer(f)
        writer.writerow(["project_id", "involvement_level", "reason"])

        for project_id in project_ids:
            logger.info(f"Processing project ID {project_id}")
            response = determine_private_sector_involvement(
                project_id, qdrant_collection=qdrant_collection
            )
            if response:
                writer.writerow(
                    [project_id, response.involvement_level, response.reason]
                )
                logger.info(f"Successfully processed project ID {project_id}")
            else:
                writer.writerow([project_id, "No data", "No data"])
                logger.warning(f"No data received for project ID {project_id}")


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
