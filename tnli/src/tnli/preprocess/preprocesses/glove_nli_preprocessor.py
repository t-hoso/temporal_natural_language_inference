import csv
from pathlib import Path
from typing import Any
import os

import numpy as np

from .preprocessor import Preprocessor


class GloveNliPreprocessor(Preprocessor):
    @classmethod
    def process(
        cls, 
        encoder,
        input_dir_name: Path,
        output_dir_name: Path, 
        model_name: str,
        max_len: int,
        data_loader
    ) -> None:
        """
        Pre-process files in given directory
        outputs processed data

        Parameters
        ----------
        input_dir_name: Path
            the input file name
            this must contain appropriate dataset
        output_dir_name: Path
            the output directory name
        model_name: str
            the model name for which this preprocess is done.
        """
        count = 1

        labels, sentence1s, sentence2s = \
            data_loader.load(input_dir_name)

        data = \
            cls.encode(
                (labels, sentence1s, sentence2s), 
                encoder, max_len=max_len
            )

        os.makedirs(output_dir_name, exist_ok=True)

        cls.write(
            data,
            output_dir_name,
            model_name,
            count
        )

    @classmethod
    def write(
        cls,
        data: tuple,
        output_dir_name: Path, 
        model_name: str,
        count: int):
        labels, sentence1s, sentence2s = data
        with open(
            str(output_dir_name/('sentence1'+".csv")), 
            encoding='utf-8', mode='w') as wf:
            writer = csv.writer(wf)
            for vector in sentence1s:
                writer.writerow(vector)
        with open(
            str(output_dir_name/('sentence2'+".csv")), 
            encoding='utf-8', mode='w') as wf:
            writer = csv.writer(wf)
            for vector in sentence2s:
                writer.writerow(vector)
        with open(
            str(output_dir_name/('labels'+'.csv')), 
            encoding='utf-8', mode='w') as wf:
            writer = csv.writer(wf)
            for label in labels:
                writer.writerow([label])

    @classmethod
    def encode(
        cls,
        data: tuple, 
        encoder: Any, 
        max_len: int
    ):
        input_vector1 = []
        input_vector2 = []
        label, sentence1_list, sentence2_list = data
        for sentence1, sentence2 in zip(sentence1_list, sentence2_list):
            if sentence1.endswith("."):
                sentence1 = sentence1[:-1]
            if sentence2.endswith("."):
                sentence2 = sentence2[:-1]
            sentence1_ids = [encoder.stoi[token] if token in encoder.stoi.keys() else encoder.stoi['unk']
                   for token in sentence1.lower().split()]
            if len(sentence1_ids) > max_len:
                sentence1_ids = sentence1_ids[:max_len]
            else:
                while len(sentence1_ids) < max_len:
                    sentence1_ids.append(0)

            sentence2_ids = [encoder.stoi[token] if token in encoder.stoi.keys() else encoder.stoi['unk']
                   for token in sentence2.lower().split()]
            if len(sentence2_ids) > max_len:
                sentence2_ids = sentence2_ids[:max_len]
            else:
                while len(sentence2_ids) < max_len:
                    sentence2_ids.append(0)
            input_vector1.append(sentence1_ids)
            input_vector2.append(sentence2_ids)
        return label, input_vector1, input_vector2

