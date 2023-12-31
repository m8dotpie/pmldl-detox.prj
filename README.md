# [PMLDL] Detoxify

**Author:** `Igor Alentev`

**Group:** `BS20-RO-01`

**Email:** `i.alentev@innopolis.university`

## Problem Definition

Can be found [here](PROBLEM.md)

## Installation and Running

**Instead of `requirenments.txt` this repository uses `conda` environment. Read further.**

I am ~~proud~~ owner of the AMD Graphics powered laptop ~~(god bless apple)~~, as a result it is nearly impossible for me to run or test anything locally. In general everything should be fine, but I was unable to test if everything runs as expected. So several issues might be possible. But I did my best to avoid any inconsistencies across the code.

- [x] All predictions preprocessed and saved [locally](data/interim/predictions/)
- [x] All metrics precalculated and saved [locally](data/interim/metrics/)
- [x] All datasets precomputed and saved [here](data/interim/dataset/) and for toxic words [here](data/interim/synonyms_dataset/)
- [x] Colab notebooks rewritten locally
- [x] [Dotenv](./.env) tuned properly
- [x] Dependencies across the `src` files as well as notebooks should work
- [x] Checkpoints provided
- [x] `conda` environment exported to environment.yml

For instance, I would recommend not running tuning and learning, rather than loading the checkpoints, which is indeed works (afaik).

## Checkpoints

It was a hard decision, but I have decided to store model checkpoints along the project itself. So if you will clone the repo, you will have to clone 0.5GB of checkpoints as well. However, it is very handy, since they are not so heavy, but useful all over the work.

## Notebooks

- Data

  - [Initial exploration](notebooks/1.0-data-exploration.ipynb)
  - [Initial preprocession](notebooks/1.1-data-preprocess.ipynb)
  - [Bad words dataset](notebooks/1.2-bad-words-dataset.ipynb)

- Metrics

  - [Introduction and draft](notebooks/2.0-metrics.ipynb)

- Baseline model

  - [Draft](notebooks/3.0-baseline-draft.ipynb)
  - [Testing](notebooks/3.1-baseline-test.ipynb)
  - [Evaluation](notebooks/3.2-baseline-evaluation.ipynb)

- Models

  - [t5-small-ft](notebooks/4.0-t5-small-ft.ipynb)
  - [t5-small-ft2](notebooks/4.1-t5-small-ft2.ipynb)

- Models Evaluation

  - [Evaluation](notebooks/5.0-evaluation.ipynb)

- Results Exploration

  - [Exploration](notebooks/6.0-results-exploration.ipynb)

## Reports

- [Proposal report](reports/first_report.pdf)

Main hypothesis, ideas and related information. The draft of the project

- [Final report](reports/final_report.pdf)

Final report, containing all the necessary information about the models, data retrieval and preprocessing, fine-tuning and evaluation

## Acknowledgements

Please do not blame me if anything does not work. I did my best to seemlessly integrate everything with each other and spent many hours on this. I am aiming at **flipped class**, so I will be very sad if I will get bad mark because of some minor issue. Even though I am all in for fair assessment and open for discussion of real issues with the work.

- Vladimir Ivanov for informative lectures
- Maxim Evgrafov and Lada Morozova for incredibly useful labs
- Skolkovo for [work](https://arxiv.org/abs/2109.08914) on detoxification
- Skolkovo for [another work](https://aclanthology.org/2022.acl-long.469/) on detoxification
- [This](https://aclanthology.org/2022.acl-long.469/) work for great showcase of metrics and transformers
- [ParaNMT-50M](https://aclanthology.org/P18-1042.pdf) dataset creators
- [Detoxify](https://github.com/unitaryai/detoxify) creators
- [WordNet](https://wordnet.princeton.edu) creators
- [Yeah Yeah Yeahs](https://music.apple.com/ru/artist/yeah-yeah-yeahs/1265171?l=en-GB) for a great music
