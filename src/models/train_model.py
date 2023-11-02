"""Module used for training the proposed model"""

import os
import sys
import warnings
import datasets
import numpy as np
import transformers
from transformers import (
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from datasets import load_dataset, load_metric, concatenate_datasets
from dotenv import find_dotenv, load_dotenv

load_dotenv()
root = os.path.dirname(find_dotenv())
sys.path.append(root)

warnings.filterwarnings("ignore")


def preprocess_function(examples, tokenizer):
    """Preprocess function for T5 model: adds prefix to source and trim sequence"""
    prefix = "make sentence non-toxic:"

    max_input_length = 256
    max_target_length = 256
    toxic = "source"
    non_toxic = "target"

    inputs = [prefix + ex if ex else " " for ex in examples[toxic]]
    targets = [ex if ex else " " for ex in examples[non_toxic]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    labels = tokenizer(targets, max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# simple postprocessing for text
def postprocess_text(preds, labels):
    """Post-processing: strip decoded predicted and label sequences"""
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels


# compute metrics function to pass to trainer
def compute_metrics(eval_preds, tokenizer, metric):
    """Compute metrics for model evaluation"""

    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
    ]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result


def train(
    checkpoint="t5-small",
    random_state=42,
    dataset=None,
    metric="sacrebleu",
    tokenizer=None,
    include_bad_words=False,
):
    """Train the model with given checkpoint on the dataset and save the results"""

    # default dataset
    if dataset is None:
        data_dir = os.getenv("DATA_DIR")
        dataset = root + "/" + data_dir + "/interim/dataset"

    transformers.set_seed(random_state)
    raw_datasets = load_dataset(dataset)

    # if we want to tune on bad words dataset as well (i.e. ft2)
    if include_bad_words:
        data_dir = os.getenv("DATA_DIR")
        dataset = root + "/" + data_dir + "/interim/synonyms_dataset"
        bad_dataset = load_dataset(dataset)
        raw_datasets = concatenate_datasets([raw_datasets, bad_dataset])

    metric = load_metric(metric)

    if tokenizer is None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(checkpoint)

    tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

    # defining the parameters for training
    batch_size = 32
    args = Seq2SeqTrainingArguments(
        f"{checkpoint}-ft",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=10,
        predict_with_generate=True,
        fp16=True,
        report_to="tensorboard",
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # instead of writing train loop we will use Seq2SeqTrainer
    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=lambda x: compute_metrics(x, tokenizer, metric),
    )

    trainer.train()
    # decide if ft2 or ft
    trainer.save_model(
        root + "/models/" + f"{checkpoint}-ft{'2' if include_bad_words else ''}"
    )
