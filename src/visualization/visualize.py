"""Visualization module used for data visualization on evaluation"""

import os
import sys
import numpy as np
import seaborn as sns
from wordcloud import WordCloud
from matplotlib import pyplot as plt

from dotenv import load_dotenv, find_dotenv


load_dotenv()
root = os.path.dirname(find_dotenv())
sys.path.append(root)


def get_wordcloud(text):
    """Generate preset wordcloud"""
    wordcloud = WordCloud(
        width=300, height=300, background_color="white", margin=0, max_font_size=40
    ).generate(text)

    return wordcloud


def plot_wordcloud(cloud, ax=None):
    """Plot the given wordcloud"""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8), dpi=100)
    ax.imshow(cloud, interpolation="bilinear")
    ax.axis("off")
    return ax


def plot_hist(
    metrics: dict,
    xlabel="Toxicity",
    ylabel="Frequency",
    plot_title="Comparison of Toxicity Distribution",
):
    """Plot histogram of given metrics"""
    titles, data = zip(*metrics.items())
    colors = ["red", "green", "skyblue"]

    for i, (title, d) in enumerate(zip(titles, data)):
        plt.hist(d, bins=15, alpha=0.5, color=colors[i], edgecolor="black", label=title)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(plot_title)
    plt.legend(loc="upper right")

    return plt


def get_violin_plot(metrics: dict):
    """Generate violin plot of given metrics"""
    fig, ax = plt.subplots(len(metrics), 1, figsize=(5, 5 * len(metrics)))
    ax = np.array(ax).flatten()

    for i, (title, d) in enumerate(metrics.items()):
        sns.violinplot(data=d, ax=ax[i], inner="quart")
        ax[i].set_title(title)

    return fig
