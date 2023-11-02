# [PMLDL] Detoxify

## Installation and Running

I am ~~proud~~ owner of the AMD Graphics powered laptop, as a result it is nearly impossible for me to run or test anything locally. In general everything should be fine, but I was unable to test if everything runs as expected. So several issues might be possible.

- [x] Colab notebooks rewritten locally
- [x] Dotenv tuned properly
- [x] Dependencies across the `src` files as well as notebooks work
- [x] Checkpoints provided

For instance, I would recommend not running tuning and learning, rather than loading the checkpoints, which is indeed works.

## Notebooks

- 1
- 2
- 3
- 4
- 5
- 6

## Reports

- [Proposal report]() - report, containing main hypothesis, ideas and related information. The draft of the project
- [Final report]() - final report, containing all the necessary information about the models, data retrieval and preprocessing, fine-tuning and evaluation

## Acknowledgements

Please do not blame me if anything does not work. I did my best to seemlessly integrate everything with each other and spent many hours on this. I am aiming at **flipped class**, so bad assignment mark will ruin me everything.

# Future references:

- [Original work](https://arxiv.org/pdf/2109.08914.pdf) about detoxification
- [Prompt tuning for detoxification](https://www.dialog-21.ru/media/5735/konodyukn120.pdf)

  - Suggests eval

- [CAE-T5 for detoxification](https://arxiv.org/pdf/2102.05456.pdf)

  - Good approach in general
  - Explains loss
  - Explains eval + uses other models

- [PARANMT-50M](https://aclanthology.org/P18-1042.pdf)

  - useful corpus, original work extracted toxic pairs somehow
