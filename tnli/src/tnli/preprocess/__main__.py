import os
from pathlib import Path

import numpy as np
import torch
import click
from transformers import BertTokenizer, RobertaTokenizer
import torchtext as text

from .preprocesses import (
    ExplainPreprocessor,
    SentenceTransformerPreprocessor,
    BertPreprocessor,
    GlovePreprocessor,
    GloveNliPreprocessor,
    ExplainNliPreprocessor
)
from .preprocesses.encoders \
    import SentenceTransformersEncoder
from .preprocesses.data_loader import(
    DataLoader,
    NliDataLoader
)

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)

DATA_DIR = os.environ.get("DATA_DIR")
MNLI_TEXT_DIR = os.environ.get("MNLI_TEXT_DIR")
SNLI_TEXT_DIR = os.environ.get("SNLI_TEXT_DIR")
SNLI_TRAIN_FILENAME = os.environ.get("SNLI_TRAIN_FILENAME")
SNLI_DEV_FILENAME = os.environ.get("SNLI_DEV_FILENAME")
SNLI_TEST_FILENAME = os.environ.get("SNLI_TEST_FILENAME")
MNLI_TRAIN_FILENAME = os.environ.get("MNLI_TRAIN_FILENAME")
MNLI_DEV_MATCHED_FILENAME = os.environ.get("MNLI_DEV_MATCHED_FILENAME")
MNLI_DEV_MISMATCHED_FILENAME = os.environ.get("MNLI_DEV_MISMATCHED_FILENAME")

MODEL_NAME_EXPLAIN_BERT = "explain_bert"
MODEL_NAME_BERT = "bert-base-uncased"
MODEL_NAME_ROBERTA = "roberta-base"
MODEL_NAME_EXPLAIN = "explain"
MODEL_NAME_DISTILLED_ROBERTA = "paraphrase-distilroberta-base-v1"
MODEL_NAME_GLOVE = "6B"
DIM_GLOVE = 300
MODEL_NAME_GLOVE_SNLI = "glove_snli"
MODEL_NAME_GLOVE_MNLI = "glove_mnli"
MODEL_NAME_EXPLAIN_SNLI = "explain_snli"
MODEL_NAME_EXPLAIN_MNLI = "explain_mnli"

MAX_LEN = 32


@click.command()
@click.argument("kind")
def run(kind):
    if kind == "mnli_explain":
        ExplainNliPreprocessor.process(
            RobertaTokenizer.from_pretrained(MODEL_NAME_ROBERTA),
            input_dir_name=Path(DATA_DIR) / MNLI_TRAIN_FILENAME,
            output_dir_name=Path(DATA_DIR) / f"{MODEL_NAME_EXPLAIN_MNLI}_train",
            model_name=MODEL_NAME_EXPLAIN_MNLI,
            max_len=MAX_LEN,
            data_loader=NliDataLoader
        )
        ExplainNliPreprocessor.process(
            RobertaTokenizer.from_pretrained(MODEL_NAME_ROBERTA),
            input_dir_name=Path(DATA_DIR) / MNLI_DEV_MATCHED_FILENAME,
            output_dir_name=Path(DATA_DIR) / f"{MODEL_NAME_EXPLAIN_MNLI}_dev_matched",
            model_name=MODEL_NAME_EXPLAIN_MNLI,
            max_len=MAX_LEN,
            data_loader=NliDataLoader
        )
        ExplainNliPreprocessor.process(
            RobertaTokenizer.from_pretrained(MODEL_NAME_ROBERTA),
            input_dir_name=Path(DATA_DIR) / MNLI_DEV_MISMATCHED_FILENAME,
            output_dir_name=Path(DATA_DIR) / f"{MODEL_NAME_EXPLAIN_MNLI}_dev_mismatched",
            model_name=MODEL_NAME_EXPLAIN_MNLI,
            max_len=MAX_LEN,
            data_loader=NliDataLoader
        )
    elif kind == "snli_explain":
        ExplainNliPreprocessor.process(
            RobertaTokenizer.from_pretrained(MODEL_NAME_ROBERTA),
            input_dir_name=Path(DATA_DIR) / SNLI_TEST_FILENAME,
            output_dir_name=Path(DATA_DIR) / f"{MODEL_NAME_EXPLAIN_SNLI}_test",
            model_name=MODEL_NAME_EXPLAIN_SNLI,
            max_len=MAX_LEN,
            data_loader=NliDataLoader
        )
        ExplainNliPreprocessor.process(
            RobertaTokenizer.from_pretrained(MODEL_NAME_ROBERTA),
            input_dir_name=Path(DATA_DIR) / SNLI_TRAIN_FILENAME,
            output_dir_name=Path(DATA_DIR) / f"{MODEL_NAME_EXPLAIN_SNLI}_train",
            model_name=MODEL_NAME_EXPLAIN_SNLI,
            max_len=MAX_LEN,
            data_loader=NliDataLoader
        )
        ExplainNliPreprocessor.process(
            RobertaTokenizer.from_pretrained(MODEL_NAME_ROBERTA),
            input_dir_name=Path(DATA_DIR) / SNLI_DEV_FILENAME,
            output_dir_name=Path(DATA_DIR) / f"{MODEL_NAME_EXPLAIN_SNLI}_dev",
            model_name=MODEL_NAME_EXPLAIN_SNLI,
            max_len=MAX_LEN,
            data_loader=NliDataLoader
        )
    elif kind == "bert":
        BertPreprocessor.process(
            BertTokenizer.from_pretrained(MODEL_NAME_BERT),
            input_dir_name=Path(DATA_DIR) / "folds",
            output_dir_name=Path(DATA_DIR) / MODEL_NAME_BERT,
            model_name=MODEL_NAME_BERT,
            max_len=MAX_LEN,
            data_loader=DataLoader
        )
    elif kind == "explain":
        ExplainPreprocessor.process(
            RobertaTokenizer.from_pretrained(MODEL_NAME_ROBERTA),
            input_dir_name=Path(DATA_DIR) / "folds",
            output_dir_name=Path(DATA_DIR) / MODEL_NAME_EXPLAIN,
            model_name=MODEL_NAME_ROBERTA,
            max_len=MAX_LEN,
            data_loader=DataLoader
        )
    elif kind == "sentence_transformer":
        SentenceTransformerPreprocessor.process(
            SentenceTransformersEncoder(MODEL_NAME_DISTILLED_ROBERTA),
            input_dir_name=Path(DATA_DIR) / "folds",
            output_dir_name=Path(DATA_DIR) / MODEL_NAME_DISTILLED_ROBERTA,
            model_name=MODEL_NAME_DISTILLED_ROBERTA,
            max_len=MAX_LEN,
            data_loader=DataLoader
        )
    elif kind == "glove":
        GlovePreprocessor.process(
            text.vocab.GloVe(name=MODEL_NAME_GLOVE, dim=DIM_GLOVE),
            input_dir_name=Path(DATA_DIR) / "folds",
            output_dir_name=Path(DATA_DIR) / "glove",
            model_name="glove",
            max_len=MAX_LEN,
            data_loader=DataLoader
        )
    elif kind == "glove_snli":
        GloveNliPreprocessor.process(
            text.vocab.GloVe(name=MODEL_NAME_GLOVE, dim=DIM_GLOVE),
            input_dir_name=Path(DATA_DIR) / SNLI_TRAIN_FILENAME,
            output_dir_name=Path(DATA_DIR) / f"{MODEL_NAME_GLOVE_SNLI}_train",
            model_name=MODEL_NAME_GLOVE_SNLI,
            max_len=MAX_LEN,
            data_loader=NliDataLoader
        )
        GloveNliPreprocessor.process(
            text.vocab.GloVe(name=MODEL_NAME_GLOVE, dim=DIM_GLOVE),
            input_dir_name=Path(SNLI_TEST_FILENAME),
            output_dir_name=Path(DATA_DIR) / f"{MODEL_NAME_GLOVE_SNLI}_test",
            model_name=MODEL_NAME_GLOVE_SNLI,
            max_len=MAX_LEN,
            data_loader=NliDataLoader
        )
        GloveNliPreprocessor.process(
            text.vocab.GloVe(name=MODEL_NAME_GLOVE, dim=DIM_GLOVE),
            input_dir_name=Path(DATA_DIR) / SNLI_DEV_FILENAME,
            output_dir_name=Path(DATA_DIR) / f"{MODEL_NAME_GLOVE_SNLI}_dev",
            model_name=MODEL_NAME_GLOVE_SNLI,
            max_len=MAX_LEN,
            data_loader=NliDataLoader
        )
    elif kind == "glove_mnli":
        GloveNliPreprocessor.process(
            text.vocab.GloVe(name=MODEL_NAME_GLOVE, dim=DIM_GLOVE),
            input_dir_name=Path(DATA_DIR) / MNLI_TRAIN_FILENAME,
            output_dir_name=Path(DATA_DIR) / f"{MODEL_NAME_GLOVE_MNLI}_train",
            model_name=MODEL_NAME_GLOVE_MNLI,
            max_len=MAX_LEN,
            data_loader=NliDataLoader
        )
        GloveNliPreprocessor.process(
            text.vocab.GloVe(name=MODEL_NAME_GLOVE, dim=DIM_GLOVE),
            input_dir_name=Path(MNLI_DEV_MATCHED_FILENAME),
            output_dir_name=Path(DATA_DIR) / f"{MODEL_NAME_GLOVE_MNLI}_dev_matched",
            model_name=MODEL_NAME_GLOVE_MNLI,
            max_len=MAX_LEN,
            data_loader=NliDataLoader
        )
        GloveNliPreprocessor.process(
            text.vocab.GloVe(name=MODEL_NAME_GLOVE, dim=DIM_GLOVE),
            input_dir_name=Path(DATA_DIR) / MNLI_DEV_MISMATCHED_FILENAME,
            output_dir_name=Path(DATA_DIR) / f"{MODEL_NAME_GLOVE_MNLI}_dev_mismatched",
            model_name=MODEL_NAME_GLOVE_MNLI,
            max_len=MAX_LEN,
            data_loader=NliDataLoader
        )
    elif kind == MODEL_NAME_EXPLAIN_BERT:
        ExplainPreprocessor.process(
            BertTokenizer.from_pretrained(MODEL_NAME_BERT),
            input_dir_name=Path(DATA_DIR) / "folds",
            output_dir_name=Path(DATA_DIR) / MODEL_NAME_EXPLAIN_BERT,
            model_name=MODEL_NAME_BERT,
            max_len=MAX_LEN,
            data_loader=DataLoader
        )

run()