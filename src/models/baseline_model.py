"""Baseline prediction model"""

import os
import re
import sys
import pandas as pd
from tqdm import tqdm
from nltk.corpus import wordnet
from dotenv import find_dotenv, load_dotenv


load_dotenv()
sys.path.append(os.path.dirname(find_dotenv()))
root = os.path.dirname(find_dotenv())


from src.data.preprocess_dataset import text_symbolic_preprocess

tqdm.pandas()


def get_nontoxic_synonym(word, blacklist):
    """Gets a non-toxic synonym for a given word from wordnet corpus"""

    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            # Check against blacklist
            if lemma.name().lower() not in blacklist:
                synonyms.add(lemma.name())

    res = list(synonyms)[0] if synonyms else None
    if res is not None:
        res = res.replace("_", " ")
    return res


def construct_replacement_dict(path=None) -> dict:
    """Construct a dictionary of toxic words and their non-toxic synonyms

    Keyword arguments:
    path -- path to the toxic words list (if None, reads env)
    """

    if path is None:
        path = (
            root
            + "/"
            + os.getenv("DATA_DIR")
            + "/external/"
            + os.getenv("TOXIC_DICT_CSV")
        )

    df = pd.read_csv(path, header=None, names=["tox"])
    df["tox"] = df["tox"].apply(text_symbolic_preprocess)
    blacklist = set(df["tox"].str.lower())

    df["ntox"] = df["tox"].apply(get_nontoxic_synonym, args=(blacklist,))
    df = df.dropna()

    return dict(zip(df["tox"], df["ntox"]))


def detoxify(sentence: str, replacement_dict: dict) -> str:
    """Replaces toxic words with their non-toxic synonyms

    Keyword arguments:
    sentence -- sentence to detoxify
    replacement_dict -- dictionary of toxic words and their non-toxic synonyms
    """

    for toxic, non_toxic in replacement_dict.items():
        # \b specifies word boundaries in regex, ensuring we're replacing whole words, not subs
        sentence = re.sub(r"\b" + re.escape(toxic) + r"\b", non_toxic, sentence)
    return sentence


def predict(df: pd.DataFrame) -> pd.DataFrame:
    """Predicts the toxicity of the translation based on the reference

    Keyword arguments:
    df -- dataframe to predict on (expected to be the output of preprocess_dataset())
    blacklist -- list of words to ignore when finding synonyms
    """

    replacement_dict = construct_replacement_dict()

    df["baseline_pred"] = df["reference"].progress_apply(
        detoxify, args=(replacement_dict,)
    )

    return df
