import csv
from pathlib import Path
from typing import Any

from .preprocessor import Preprocessor


MAX_LEN = 32
PREFIX_LENGTH = "length_"
PREFIX_IDS = "ids_"


class ExplainPreprocessor(Preprocessor):
    @classmethod
    def write(
        cls,
        data: tuple,
        output_dir_name: Path, 
        model_name: str,
        count: int):
        lengths, encoded_ids = data
        with open(
            str(output_dir_name/(PREFIX_LENGTH+str(count)))+".csv", 
            encoding="utf-8", mode="w") as wf:
            writer = csv.writer(wf)
            for length in lengths:
                writer.writerow([length])
        with open(
            str(output_dir_name/(PREFIX_IDS+str(count)))+".csv", 
            encoding="utf-8", mode="w") as wf:
            writer = csv.writer(wf)
            for ids in encoded_ids:
                writer.writerow(ids)

    @classmethod
    def encode(
        cls,
        data: tuple, 
        encoder: Any, 
        max_len: int
    ):
        """
        encode one set of data

        Parameters
        ----------
        sentence1_list: np.ndarray
            the list of sentence1 of dataset
        sentence2_list: np.ndarray
            the list of sentence2 of dataset
        encoder: Any
            the encoder such as BertTokenizer
        max_len: int
            the maximum length of input
        """
        _, sentence1_list, sentence2_list = data
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
        return input_length_list, input_id_list

