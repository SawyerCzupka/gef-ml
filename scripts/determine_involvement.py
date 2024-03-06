import logging

from gef_ml.inference import determine_private_sector_involvement
from gef_ml.utils.log_config import setup_logging

setup_logging()

logger = logging.getLogger(__name__)


def main():
    project_id = "1234"

    response = determine_private_sector_involvement(project_id)

    if response:
        logger.info(f"Private sector involvement: {response.involvement_level}")
        logger.info(f"Reason: {response.reason}")


if __name__ == "__main__":
    main()
