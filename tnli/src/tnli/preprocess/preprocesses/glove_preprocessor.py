import csv
from pathlib import Path
from typing import Any

import numpy as np
import torchtext as text

from .preprocessor import Preprocessor


class GlovePreprocessor(Preprocessor):
    @classmethod
    def write(
        cls,
        data: tuple,
        output_dir_name: Path, 
        model_name: str,
        count: int):
        sentence1s, sentence2s = data
        with open(
            str(output_dir_name/('sentence1_'+str(count)))+".csv", 
            encoding='utf-8', mode='w') as wf:
            writer = csv.writer(wf)
            for sentence in sentence1s:
                writer.writerow(sentence)
        with open(
            str(output_dir_name/('sentence2_'+str(count)))+".csv", 
            encoding='utf-8', mode='w') as wf:
            writer = csv.writer(wf)
            for sentence in sentence2s:
                writer.writerow(sentence)

    @classmethod
    def encode(
        cls,
        data: tuple, 
        encoder: Any, 
        max_len: int
    ):
        _, sentence1_list, sentence2_list = data
        return (
            cls.__encode_sentence(
                data[0],
                encoder,
                max_len
            ),
            cls.__encode_sentence(
                data[1],
                encoder,
                max_len
            )

        )
    
    @staticmethod
    def __encode_sentence(
        sentences: np.ndarray,
        encoder: text.vocab.GloVe,
        max_len: int
    ):
        """
        encodes glove
        """
        encoded_sentences = []
        for sentence in sentences:
            if sentence.endswith('.'):
                sentence = sentence[:-1]
            ids = [
                encoder.stoi[token] if token in encoder.stoi.keys() \
                    else encoder.stoi["unk"]
                   for token in sentence.lower().split()]
            if len(ids) > max_len:
                ids = ids[:max_len]
            else:
                while len(ids) < max_len:
                    ids.append(0)
            encoded_sentences.append(ids)
        return encoded_sentences
