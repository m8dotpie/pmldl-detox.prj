import numpy as np
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud
from matplotlib import pyplot as plt

from dotenv import load_dotenv, find_dotenv
import os
import sys

load_dotenv()
root = os.path.dirname(find_dotenv())
sys.path.append(root)


def get_wordcloud(text):
    wordcloud = WordCloud(
        width=300, height=300, background_color="white", margin=0, max_font_size=40
    ).generate(text)

    return wordcloud


def plot_wordcloud(cloud, ax=None):
    """Plot the given wordcloud"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
    ax.imshow(cloud, interpolation="bilinear")
    ax.axis("off")
    return ax


def get_violin_plot():
    pass
