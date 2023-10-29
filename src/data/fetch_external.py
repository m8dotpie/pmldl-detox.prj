"""Fetch heavy external resources"""

import requests
from dotenv import find_dotenv, load_dotenv
import os

load_dotenv()
root = os.path.dirname(find_dotenv())


def fetch_embeddings(url=None):
    """Fetches the embeddings from the web and saves them locally."""

    if url is None:
        url = "http://nlp.uoregon.edu/download/embeddings/glove.6B.100d.txt"

    filename = root + "/" + os.getenv("DATA_DIR") + "/external/" + "embeddings.txt"
    response = requests.get(url, timeout=100)

    if response.ok:
        with open(filename, "wb") as f:
            f.write(response.content)


def fetch_badwords(url=None):
    """Fetches the badwords from the web and saves them locally."""

    if url is None:
        url = "https://gist.githubusercontent.com/m8dotpie/b2d9afcacb1c447c5dc4d509f7007470/raw/c18ccc661ee83fba81a1522543cc5450c276971c/bad_words.csv"

    filename = root + "/" + os.getenv("DATA_DIR") + "/external/" + "bad-words.csv"
    response = requests.get(url, timeout=100)

    if response.ok:
        with open(filename, "wb") as f:
            f.write(response.content)
