import os
import csv
import requests
import logging
from bs4 import BeautifulSoup
from dask.delayed import delayed
from dask.base import compute
from urllib.parse import urlparse

from .project_ids import gef6_project_ids, project_ids

PROJECTS_CSV_PATH = "projects.csv"
BASE_URL = "https://www.thegef.org/projects-operations/projects/"
OUTPUT_PATH = "../data/gef-6"
INTERESTED_YEARS = [i for i in range(2012, 2024)]

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def get_project_ids_from_csv(path):
    project_ids = []
    with open(path, "r") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header row
        for row in reader:
            try:
                year = int(row[9])
                if year in INTERESTED_YEARS:
                    project_id = row[1]
                    project_ids.append(project_id)
            except ValueError:
                continue
    return project_ids


def download_pdfs_from_project_page(project_id):
    url = BASE_URL + str(project_id)
    response = requests.get(url)
    if response.status_code != 200:
        logging.warning(f"Failed to access website: {url}")
        return [], []

    soup = BeautifulSoup(response.text, "html.parser")
    links = soup.find_all("a")

    # List of valid file extensions
    valid_extensions = [".pdf", ".doc", ".docx", ".txt"]

    idx = 0

    downloaded_files = []
    skipped_extensions = set()
    for link in links:
        href = link.get("href")
        if href:
            file_extension = os.path.splitext(href)[1]
            if any(file_extension.endswith(ext) for ext in valid_extensions):
                try:
                    parsed_url = urlparse(href)
                    if parsed_url.scheme == "" or parsed_url.netloc == "":
                        logging.warning(f"Skipping invalid URL: {href}")
                        continue

                    file_response = requests.get(href)
                    file_path = os.path.join(
                        OUTPUT_PATH,
                        f"{project_id}/p{project_id}_doc{idx}{file_extension}",
                    )

                    # Make project dir
                    if not os.path.exists(os.path.join(OUTPUT_PATH, f"{project_id}")):
                        os.makedirs(os.path.join(OUTPUT_PATH, f"{project_id}"))

                    idx += 1
                    with open(file_path, "wb") as file:
                        file.write(file_response.content)
                    logging.info(f"Downloaded file: {file_path}")
                    downloaded_files.append(file_path)
                except Exception as e:
                    logging.error(f"Failed to download file from: {href}. Error: {e}")
            else:
                skipped_extensions.add(file_extension)

    logging.warning(f"Skipped extensions: {skipped_extensions}")

    return downloaded_files, skipped_extensions


def main():
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    tasks = [delayed(download_pdfs_from_project_page)(pid) for pid in gef6_project_ids]
    _ = compute(*tasks)


if __name__ == "__main__":
    main()
