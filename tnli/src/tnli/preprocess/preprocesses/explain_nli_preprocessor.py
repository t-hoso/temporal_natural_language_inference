import csv
from pathlib import Path
from typing import Any
import os

import numpy as np

from .preprocessor import Preprocessor


class ExplainNliPreprocessor(Preprocessor):
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
        labels, input_length_list, input_id_list = data

        with open(str(output_dir_name/('length_'+str(count)))+".csv", encoding='utf-8', mode='w') as wf:
            writer = csv.writer(wf)
            for length in input_length_list:
                writer.writerow([length])
        with open(str(output_dir_name/('ids_'+str(count)))+".csv", encoding='utf-8', mode='w') as wf:
            writer = csv.writer(wf)
            for ids in input_id_list:
                writer.writerow(ids)
        with open(str(output_dir_name/('labels_'+str(count)))+'.csv', encoding='utf-8', mode='w') as wf:
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
        labels, sentence1_list, sentence2_list = data
        input_length_list = []
        input_id_list = []
        for sentence1, sentence2 in zip(sentence1_list, sentence2_list):
            if sentence1.endswith("."):
                sentence1 = sentence1[:-1]
            if sentence2.endswith("."):
                sentence2 = sentence2[:-1]
            sentence1_input_ids = encoder.encode(sentence1, add_special_tokens=False)
            sentence2_input_ids = encoder.encode(sentence2, add_special_tokens=False)
            input_ids = sentence1_input_ids + [2] + sentence2_input_ids
            if len(input_ids) > max_len - 2:
                input_ids = input_ids[:max_len - 2]
            length = len(input_ids) + 2
            input_ids = [0] + input_ids + [2]
            input_length_list.append(length)
            input_id_list.append(input_ids)
        return labels, input_length_list, input_id_list

