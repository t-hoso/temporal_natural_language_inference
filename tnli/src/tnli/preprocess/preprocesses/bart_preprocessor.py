import csv
from pathlib import Path
from typing import Any

import numpy as np

from .preprocessor import Preprocessor


class BartPreprocessor(Preprocessor):
    @classmethod
    def write(
        cls,
        data: tuple,
        output_dir_name: Path, 
        model_name: str,
        count: int):
        data
        with open(
            str(output_dir_name/('input_ids_'+str(count)))+".csv", 
            encoding='utf-8', mode='w') as wf:
            writer = csv.writer(wf)
            for ids in data:
                writer.writerow(
                    ids['input_ids'].numpy().tolist()[0]
                )
        # with open(
        #     str(output_dir_name/('token_type_ids_'+str(count)))+".csv", 
        #     encoding='utf-8', mode='w') as wf:
        #     writer = csv.writer(wf)
        #     for ids in data:
        #         writer.writerow(
        #             ids['token_type_ids'].numpy().tolist()[0]
        #         )
        with open(
            str(output_dir_name/('attention_mask_'+str(count)))+".csv", 
            encoding='utf-8', mode='w') as wf:
            writer = csv.writer(wf)
            for masks in data:
                writer.writerow(
                    masks["attention_mask"].numpy().tolist()[0]
                )

    @classmethod
    def encode(
        cls,
        data: tuple, 
        encoder: Any, 
        max_len: int
    ):
        _, sentence1_list, sentence2_list = data
        encoded_inputs = []
        for sentence1, sentence2 in zip(sentence1_list, sentence2_list):
            formatted_pair = f"{sentence1} </s><s> {sentence2}"
            encoded_inputs.append(
                encoder.encode_plus(
                    formatted_pair, 
                    return_tensors='pt', 
                    pad_to_max_length=True, 
                    truncation=True,
                    max_length=max_len
                )
            )
        return encoded_inputs
