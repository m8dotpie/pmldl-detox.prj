"""Preprocessing module for the dataset."""

import re
import nltk
import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)


def text_symbolic_preprocess(text: str) -> str:
    """Symbolic preprocess of text (i.e. remove punctuation, etc)"""

    fix = text
    fix = re.sub(r"\s+", " ", fix)
    fix = re.sub(r"\d+", " ", fix)
    fix = re.sub(r"([.!?])", r" ", fix)
    fix = re.sub(r"[^a-zA-Z.!?]+", r" ", fix)
    fix = fix.strip()
    fix = fix.lower()

    return fix


def text_semantic_preprocess(text: str) -> str:
    """Semantic preprocess of text (lemmatization)"""

    fix = text
    fix = word_tokenize(fix)
    fix = [word for word in fix if word not in stopwords.words("english")]
    fix = [WordNetLemmatizer().lemmatize(word) for word in fix]
    return fix


def text_difference_preprocess(row) -> str:
    """
    Difference preprocess of text
    This function assumed to receive a row with columns 't1', 't2'
    and returns a pair of strings --- symmetric difference of the two texts
    """
    text1, text2 = row["t1"], row["t2"]
    new_text1 = [word for word in text1 if word not in text2]
    new_text2 = [word for word in text2 if word not in text1]

    return new_text1, new_text2


def dataframe_preprocess(
    df: pd.DataFrame,
    symbolic=True,
    semantic=False,
    difference=False,
    merge_after=False,
    mask=None,
    df_max_len=100,
    random_state=42,
) -> pd.DataFrame:
    """Preprocess dataframe with 2 optional stages

    Keyword arguments:
    df -- dataframe to preprocess (expected to be the output of make_dataset())
    symbolic -- whether to perform symbolic preprocessing (default True)
    semantic -- whether to perform semantic preprocessing (default False)
    difference -- whether to perform difference preprocessing (default False)
    merge_after -- whether to merge the texts after preprocessing (default False)
    mask -- mask to apply to the dataframe (default None)
    df_max_len -- maximum length of the dataframe (default 100)
    random_state -- random state for sampling (default 42)
    """

    in_mask = df["trn_tox"] > df["ref_tox"]
    temp = df.loc[in_mask, "reference"].copy()
    df.loc[in_mask, "reference"] = df.loc[in_mask, "translation"]
    df.loc[in_mask, "translation"] = temp

    df["t1"] = df["reference"]
    df["t2"] = df["translation"]

    if mask is None:
        mask = (df["ref_tox"] > 0.99) & (df["trn_tox"] < 0.01) | (
            df["trn_tox"] > 0.99
        ) & (df["ref_tox"] < 0.01)

    df = df[mask]

    if (df_max_len is not None) and (len(df) > df_max_len):
        df = df.sample(df_max_len, random_state=random_state)

    if symbolic:
        df["t1"] = df["t1"].apply(text_symbolic_preprocess)
        df["t2"] = df["t2"].apply(text_symbolic_preprocess)

    if semantic:
        df["t1"] = df["t1"].apply(text_semantic_preprocess)
        df["t2"] = df["t2"].apply(text_semantic_preprocess)

    if difference:
        df[["t1", "t2"]] = df[["t1", "t2"]].apply(
            text_difference_preprocess, axis=1, result_type="expand"
        )

    if merge_after:
        df["t1"] = df["t1"].str.join(" ")
        df["t2"] = df["t2"].str.join(" ")

    return df
