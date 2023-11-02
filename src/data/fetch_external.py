"""Fetch heavy external resources"""

import os
import requests
from dotenv import find_dotenv, load_dotenv


load_dotenv()
root = os.path.dirname(find_dotenv())


def fetch_embeddings(url=None):
    """Fetches the embeddings from the web and saves them locally."""

    if url is None:
        url = os.getenv("GLOVE_URL")

    filename = root + "/" + os.getenv("DATA_DIR") + "/external/" + "embeddings.txt"
    response = requests.get(url, timeout=100)

    if response.ok:
        with open(filename, "wb") as f:
            f.write(response.content)


def fetch_badwords(url=None):
    """Fetches the badwords from the web and saves them locally."""

    if url is None:
        url = os.getenv("BADWORDS_URL")

    filename = root + "/" + os.getenv("DATA_DIR") + "/external/" + "bad-words.csv"
    response = requests.get(url, timeout=100)

    if response.ok:
        with open(filename, "wb") as f:
            f.write(response.content)
