import os
import csv
import requests
import logging
from bs4 import BeautifulSoup
import dask
from dask import delayed
from urllib.parse import urlparse

PROJECTS_CSV_PATH = "projects.csv"
BASE_URL = "https://www.thegef.org/projects-operations/projects/"
OUTPUT_PATH = "dump"
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
                        OUTPUT_PATH, f"file{project_id}_{idx}{file_extension}"
                    )
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

    # project_ids = get_project_ids_from_csv(PROJECTS_CSV_PATH)
    project_ids = [
        6915,
        6930,
        6940,
        6943,
        6944,
        6945,
        6947,
        6949,
        6955,
        6960,
        6962,
        6964,
        6966,
        6970,
        6971,
        6973,
        6980,
        6982,
        6991,
        7993,
        8005,
        8015,
        8023,
        9037,
        9045,
        9048,
        9088,
        9095,
        9103,
        9112,
        9114,
        9121,
        9124,
        9125,
        9135,
        9137,
        9139,
        9143,
        9160,
        9163,
        9167,
        9172,
        9173,
        9185,
        9196,
        9199,
        9211,
        9220,
        9226,
        9240,
        9273,
        9276,
        9282,
        9289,
        9309,
        9314,
        9319,
        9320,
        9334,
        9335,
        9339,
        9340,
        9341,
        9342,
        9350,
        9351,
        9352,
        9354,
        9359,
        9360,
        9365,
        9367,
        9369,
        9379,
        9390,
        9391,
        9416,
        9451,
        9453,
        9454,
        9455,
        9457,
        9460,
        9467,
        9480,
        9486,
        9495,
        9502,
        9511,
        9529,
        9533,
        9535,
        9547,
        9548,
        9567,
        9622,
        9634,
        9635,
        9641,
        9644,
        9651,
        9658,
        9666,
        9674,
        9675,
        9690,
        9700,
        9707,
        9712,
        9720,
        9724,
        9731,
        9734,
        9739,
        9741,
        9755,
        9795,
        9803,
        9804,
        9807,
        9813,
        9814,
        9817,
        9820,
        9821,
        9826,
        9828,
        9829,
        9833,
        9834,
        9835,
        9837,
        9840,
        9861,
        9864,
        9865,
        9889,
        9917,
        9923,
        9928,
        9931,
        9941,
        9947,
        9949,
        9950,
        9959,
        9991,
        9993,
        10023,
        10029,
        10042,
        10051,
        10054,
        10075,
        10079,
        10082,
        10085,
        10086,
        10087,
        10094,
        10103,
        10108,
        10110,
        10113,
        10117,
        10118,
        10121,
        10122,
        10123,
        10124,
        10125,
        10138,
        10142,
        10143,
        10149,
        10151,
        10155,
        10157,
        10163,
        10165,
        10172,
        10184,
        10186,
        10188,
        10195,
        10202,
        10204,
        10220,
        10227,
        10228,
        10232,
        10234,
        10235,
        10236,
        10239,
        10246,
        10249,
        10250,
        10251,
        10252,
        10253,
        10254,
        10257,
        10260,
        10264,
        10265,
        10267,
        10269,
        10270,
        10279,
        10281,
        10282,
        10287,
        10293,
        10308,
        10316,
        10317,
        10318,
        10321,
        10322,
        10343,
        10344,
        10352,
        10353,
        10356,
        10358,
        10360,
        10365,
        10368,
        10373,
        10374,
        10394,
        10401,
        10412,
        10425,
        10427,
        10428,
        10429,
        10432,
        10435,
        10436,
        10437,
        10442,
        10443,
        10446,
        10449,
        10452,
        10456,
        10464,
        10466,
        10467,
        10472,
        10479,
        10504,
        10510,
        10551,
        10581,
        10592,
        10596,
        10609,
        10617,
        10622,
        10623,
        10625,
        10626,
        10628,
        10632,
        10637,
        10655,
        10667,
        10746,
        10757,
        10798,
        10829,
        10918,
        11022,
    ]
    tasks = [delayed(download_pdfs_from_project_page)(pid) for pid in project_ids]
    _ = dask.compute(*tasks)


if __name__ == "__main__":
    main()
