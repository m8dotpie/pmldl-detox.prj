# TODO:

- Write docstrings
- Discriminator dataset
- ~~script for downloading glove and bad-words~~
- inference and learning time graphs

# Info

- Rust compiler might be required for detoxify
- pkg config on ubuntu

# Metrics

- BLEU
- METEOR
- Semantic Similarity (glove, word2vec)
- Discriminator (the better the text, the less likely it is detected as machine generated)

# References

- Bad words: <https://www.kaggle.com/datasets/nicapotato/bad-bad-words/>
- Detoxify: <https://github.com/unitaryai/detoxify>
- wordcloud: <https://amueller.github.io/word_cloud/>
- docstring: <https://peps.python.org/pep-0257/>
- translate: <https://huggingface.co/docs/transformers/tasks/translation>
- bart-base-detox: <https://huggingface.co/s-nlp/bart-base-detox?text=fuck+you>
