"""Make the dataset from the filtered_paranmt.zip file"""

import os
import zipfile
from dotenv import find_dotenv, load_dotenv
import requests

load_dotenv()
root = os.path.dirname(find_dotenv())


def fetch_tsv(
    path=None,
):
    """Fetches the filtered_paranmt.zip file from the given path

    Keyword arguments:
    path -- path to the filtered_paranmt.zip file. (if None, reads env)
    """

    if path is None:
        path = os.getenv("PARANMT_URL")

    filename = root + "/" + os.getenv("DATA_DIR") + "/raw/" + os.getenv("PARANMT_ZIP")
    response = requests.get(path, timeout=100)

    if response.ok:
        with open(filename, "wb") as f:
            f.write(response.content)


def unzip_tsv(path=None):
    """Unzips the filtered_paranmt.zip file

    Keyword arguments:
    path -- path to the filtered_paranmt.zip file (if None, constructs default)
    """

    data_dir = os.getenv("DATA_DIR")
    paranmt = os.getenv("PARANMT_ZIP")

    if path is None:
        path = root + "/" + data_dir + "/raw/" + paranmt

    with zipfile.ZipFile(path, "r") as zip_ref:
        zip_ref.extractall(root + "/" + data_dir + "/raw/" + paranmt[:-4])


def cleanup(path=None):
    """Removes the zip file

    Keyword arguments:
    path -- path to the zip file (if None, constructs default)
    """

    data_dir = os.getenv("DATA_DIR")
    paranmt = os.getenv("PARANMT_ZIP")

    if path is None:
        path = root + "/" + data_dir + "/raw/" + paranmt

    os.remove(path)


def make_dataset():
    """Make the dataset from the filtered_paranmt.zip file"""
    fetch_tsv()
    unzip_tsv()
    cleanup()
