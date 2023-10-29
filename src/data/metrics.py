"""Evaluation metrics methods"""

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import single_meteor_score
from nltk.tokenize import word_tokenize

from detoxify import Detoxify

import os
import sys
from dotenv import load_dotenv, find_dotenv

load_dotenv()
root = os.path.dirname(find_dotenv())

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import KeyedVectors


def get_sentence_vector(sentence, model):
    """Computes the vector representation of a sentence"""

    words = sentence.split()
    # Filter out words that are not in the model's key_to_index
    words = [word for word in words if word in model.key_to_index]
    # If no words of the sentence are in the model's key_to_index, return a zero vector
    if len(words) == 0:
        return np.zeros(model.vector_size)
    # Get vectors for each word and average them
    vectors = [model[word] for word in words]
    sentence_vector = np.mean(vectors, axis=0)
    return sentence_vector


def compute_semantic_similarity(sent1, sent2, model):
    """Computes cosine similarity between two sentences"""

    vec1 = get_sentence_vector(sent1, model)
    vec2 = get_sentence_vector(sent2, model)
    return cosine_similarity([vec1], [vec2])[0][0]


def cosine_similarity_score(reference: list, hypothesis: list) -> float:
    """Computes cosine similarity score"""

    data_dir = os.getenv("DATA_DIR")
    embedding = os.getenv("EMBEDDING")

    glove_model = KeyedVectors.load_word2vec_format(
        root + "/" + data_dir + "/external/" + embedding, binary=False, no_header=True
    )

    scores = []
    for ref, hyp in zip(reference, hypothesis):
        scores.append(compute_semantic_similarity(ref, hyp, glove_model))

    return scores, sum(scores) / len(scores)


def tokenize(sentence: str) -> list:
    """Tokenizes a sentence"""
    return word_tokenize(sentence)


def _compute_bleu_score(reference, hypothesis):
    # Tokenize the sentences
    reference_tokens = [tokenize(reference)]
    hypothesis_tokens = tokenize(hypothesis)

    # Use smoothing function
    smoothie = SmoothingFunction().method4

    return sentence_bleu(
        reference_tokens, hypothesis_tokens, smoothing_function=smoothie
    )


def blue_score(reference: list, hypothesis: list) -> float:
    """Computes BLEU score"""

    scores = []
    for ref, hyp in zip(reference, hypothesis):
        scores.append(_compute_bleu_score(ref, hyp))

    return scores, sum(scores) / len(scores)


def _compute_meteor_score(reference, hypothesis):
    # Tokenize the sentences
    reference_tokens = word_tokenize(reference)
    hypothesis_tokens = word_tokenize(hypothesis)

    # The METEOR function in nltk expects the reference as a string and the hypothesis as a list of tokens
    return single_meteor_score(reference_tokens, hypothesis_tokens)


def meteor_score(reference: list, hypothesis: list) -> float:
    """Computes METEOR score"""

    scores = []
    for ref, hyp in zip(reference, hypothesis):
        scores.append(_compute_meteor_score(ref, hyp))

    return scores, sum(scores) / len(scores)


def toxicity_score(samples: list, batch_size=25) -> float:
    """Computes toxicity score"""

    scores = []
    for i in range(0, len(samples), batch_size):
        batch = samples[i : i + batch_size]
        scores += Detoxify("original").predict(batch)["toxicity"]
    score = sum(scores) / len(scores)

    return scores, 1 - score
