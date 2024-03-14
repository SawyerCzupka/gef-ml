import csv
import logging

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

from gef_ml.inference import determine_private_sector_involvement
from gef_ml.utils.log_config import setup_logging

setup_logging()

logger = logging.getLogger(__name__)

GEF_6_COLLECTION = "gef_6_512_64"

OUTPUT_CSV = "involvement_results_v2.csv"
EXCEL_PATH = "../data/ieo_private_sector_analysis.xlsx"
EXCEL_SHEET = "projectlist"


def determine_involvement_batch(
    project_ids: list[str], qdrant_collection: str = GEF_6_COLLECTION
):

    logger.info(f"Processing {len(project_ids)} project IDs")
    path = "../data/" + OUTPUT_CSV
    with open(path, "w", newline="") as f:
        logger.info(f"Writing results to {path}")
        writer = csv.writer(f)
        writer.writerow(
            [
                "project_id",
                "involvement_level",
                "secondary_involvement_level",
                "reason",
                "extra_info",
            ]
        )

        for project_id in tqdm(project_ids, desc="Processing projects"):
            logger.info(f"Processing project ID {project_id}")
            response = determine_private_sector_involvement(
                project_id, qdrant_collection=qdrant_collection
            )
            if response:
                writer.writerow(
                    [
                        project_id,
                        response.involvement_level,
                        response.secondary_involvement_level,
                        response.reason,
                        response.extra_info,
                    ]
                )
                f.flush()
                logger.info(f"Successfully processed project ID {project_id}")
            else:
                writer.writerow([project_id, "No data", "No data"])
                logger.warning(f"No data received for project ID {project_id}")


def get_project_ids_from_xlsx(gef_phase: int) -> list[str]:
    df = pd.read_excel(io=EXCEL_PATH, sheet_name=EXCEL_SHEET)
    df["GEF ID"] = df["GEF ID"].astype(str)

    gef_6 = df[df["GEF Phase"].str.contains(f"GEF - {gef_phase}")]

    return gef_6["GEF ID"].tolist()


def main():
    # collection = GEF_6_COLLECTION
    collection = "gef_6_1024_96"

    gef_6_ids = get_project_ids_from_xlsx(gef_phase=6)
    determine_involvement_batch(project_ids=gef_6_ids, qdrant_collection=collection)


if __name__ == "__main__":

    main()
